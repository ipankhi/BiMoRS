import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
import argparse

# Check if GPU is available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load models and tokenizers to the GPU if available
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

def normalize_image_tensor(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def preprocess_image_tensor(image_tensor):
    normalized_tensor = normalize_image_tensor(image_tensor)
    to_pil = transforms.ToPILImage()
    normalized_tensor = to_pil(normalized_tensor)
    pixel_values = processor(images=normalized_tensor, return_tensors="pt").to(device)
    return pixel_values

def extract_features_and_reshape(caption, bert_tokenizer, bert_model):
    inputs = bert_tokenizer(caption, return_tensors="pt").to(device)
    outputs = bert_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = torch.mean(last_hidden_state, dim=1)
    return pooled_output

def generate_caption(image_tensor, model, processor):
    inputs = preprocess_image_tensor(image_tensor)
    pixel_values = inputs['pixel_values']
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

def get_reshaped_tensor_from_caption(image_tensor):
    caption = generate_caption(image_tensor, model, processor)
    reshaped_tensor = extract_features_and_reshape(caption, bert_tokenizer, bert_model)
    return reshaped_tensor

def process_and_save_tensors(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    for class_name in os.listdir(input_folder):
        class_input_path = os.path.join(input_folder, class_name)
        class_output_path = os.path.join(output_folder, class_name)

        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        for img_name in os.listdir(class_input_path):
            img_path = os.path.join(class_input_path, img_name)
            output_tensor_path = os.path.join(class_output_path, img_name.replace('.jpg', '.pt'))
            print(output_tensor_path)
            if not os.path.isfile(img_path):
                continue

            # Check if the output tensor file exists and is valid
            if os.path.exists(output_tensor_path):
                try:
                    if os.path.getsize(output_tensor_path) > 0:
                        # Attempt to load the file to check for corruption
                        torch.load(output_tensor_path)
                        print(f"Caption file for {img_name} is valid. Skipping...")
                        continue
                    else:
                        print(f"Caption file for {img_name} is empty. Reprocessing...")
                except Exception as e:
                    print(f"Caption file for {img_name} is corrupted. Reprocessing... ({e})")

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).to(device)
                reshaped_tensor = get_reshaped_tensor_from_caption(image_tensor)
                torch.save(reshaped_tensor.cpu(), output_tensor_path)
                print(f"Saved tensor for {img_name} to {output_tensor_path}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

# if __name__ == "__main__":
#     input_folder = "/home/sunny/Pankhi/Ours/DATA/PatternNetv2/images"
#     output_folder = "/home/sunny/Pankhi/Ours/DATA/PatternNetv2/caption_images"
#     process_and_save_tensors(input_folder, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to generate captions and extract CLIP features.")
    parser.add_argument("input_folder", type=str, help="Path to the input image folder")
    parser.add_argument("output_folder", type=str, help="Path to save output caption features")
    
    args = parser.parse_args()

    process_and_save_tensors(args.input_folder, args.output_folder)