import os
import os.path as osp
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .utils import cls_token_gen as token_gen
from .utils import kn_token_gen as kn_gen

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        x = x @ self.text_projection
        return x


def exists(val):
    return val is not None


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim else None

        self.norm = self.norm.to(torch.float16)
        if self.norm_context:
            self.norm_context = self.norm_context.to(torch.float16)
        self.to(torch.float16)

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = x_q.to(torch.float16)
        x_q = self.norm(x_q)

        if x_kv is not None:
            x_kv = x_kv.to(torch.float16)
            if self.norm_context:
                x_kv = self.norm_context(x_kv)
            else:
                x_kv = x_q
        else:
            x_kv = x_q

        self.fn = self.fn.to(torch.float16)
        return self.fn(x_q, x_kv, x_kv, **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, latent_dim, kv_dim, cross_heads=4, seq_dropout_prob=0.):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.cross_attend_blocks = PreNorm(
            latent_dim,
            nn.MultiheadAttention(
                latent_dim,
                num_heads=cross_heads,
                kdim=kv_dim,
                vdim=kv_dim,
                dropout=seq_dropout_prob,
                batch_first=True
            ),
            context_dim=kv_dim
        )

    def forward(self, data, prompt, mask=None):
        data = data.to(torch.float16)
        prompt = prompt.to(torch.float16)

        kv, _ = self.cross_attend_blocks(prompt, data, key_padding_mask=mask)
        prompt = prompt + kv
        return prompt


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(768, 768 // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(768 // 16, ctx_dim))
        ]))

        self.gap = nn.AdaptiveAvgPool2d((1, 512))

        self.cross_attn = CrossAttention(
            latent_dim=ctx_dim,
            kv_dim=ctx_dim,
            cross_heads=4
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, img_emb, caption):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx

        ctx_prompt = ctx.unsqueeze(0).expand(img_emb.shape[0], -1, -1)

        meta_output = self.meta_net(img_emb)
        pooled_meta_output = self.gap(meta_output)

        key_prompt = torch.cat([pooled_meta_output, caption], dim=1)

        ctx_shifted = self.cross_attn(key_prompt, ctx_prompt)

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.token_gen = token_gen()
        self.kn_gen = kn_gen()

    def forward(self, image, caption, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image = image.to(self.dtype)
        image_features, _, img_embed = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        caption = self.token_gen(caption)
        caption = caption.to(self.dtype).to(img_embed.device)

        prompts = self.prompt_learner(img_embed, caption)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label, caption = self.parse_batch_train(batch)

        if self.cfg.TRAINER.COCOOP.PREC == "amp":
            with autocast():
                loss = self.model(image, caption, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = self.model(image, caption, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        torch.cuda.empty_cache()
        return {"loss": loss.item()}

    def parse_batch_train(self, batch):
        impath_list = batch["impath"]
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        captions = []
        for impath in impath_list:
            caption_path = impath.replace("/images/", "/caption_images/").replace(".jpg", ".pt")

            if not os.path.exists(caption_path):
                raise FileNotFoundError(f"Caption file does not exist: {caption_path}")
            if os.path.getsize(caption_path) == 0:
                raise ValueError(f"Caption file is empty: {caption_path}")

            caption = torch.load(caption_path).to(self.device)
            captions.append(caption)

        captions = torch.stack(captions)
        return image, label, captions

    def after_epoch(self):
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result

            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or (self.epoch + 1) == self.max_epoch:
            self.save_model(self.epoch, self.output_dir)

    def load_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image = Image.open(image_path).convert('RGB')
        return transform(image)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("No pretrained model given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)

            print(f"Loading weights to {name} from {model_path}")
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        split = split or self.cfg.TEST.SPLIT
        print(f"Evaluate on the *{split}* set")

        for batch in tqdm(self.test_loader):
            image, label, caption = self.parse_batch_test(batch)
            logits = self.model_inference(image, caption, label)
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)

        return list(results.values())[0]
