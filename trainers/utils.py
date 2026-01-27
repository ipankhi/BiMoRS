import torch
import torch.nn as nn


class HighFrequencyExtractor(nn.Module):
    """
    This class is used to extract high-frequency components from images.
    """
    def __init__(self, save_path="./high_freq_images", radius_ratio=0.25):
        super(HighFrequencyExtractor, self).__init__()
        self.save_path = save_path
        self.radius_ratio = radius_ratio
        os.makedirs(save_path, exist_ok=True)

    def apply_fourier_transform(self, image):
        # Apply 2D Fourier Transform
        fourier_transform = torch.fft.fft2(image)
        return fourier_transform

    def apply_inverse_fourier_transform(self, high_frequencies):
        # Apply the Inverse Fourier Transform to obtain the high-frequency components in spatial domain
        ift_image = torch.fft.ifft2(high_frequencies)
        return torch.abs(ift_image)

    def get_high_frequencies(self, fourier_transform):
        # Isolate high frequencies (typically, high frequencies are found at the corners of the FFT result)
        _, height, width = fourier_transform.shape
        center_height, center_width = height // 2, width // 2
        radius = int(min(center_height, center_width) * self.radius_ratio)  # Define radius for high frequency area

        # Create a mask with low frequencies set to 0
        mask = torch.ones_like(fourier_transform)
        mask[:, center_height - radius:center_height + radius, center_width - radius:center_width + radius] = 0

        # Apply mask to get high frequencies
        high_frequencies = fourier_transform * mask

        return high_frequencies

    def forward(self, images):
        high_freq_images = []
        for index, image in enumerate(images):
            # Apply Fourier Transform
            fourier_transform = self.apply_fourier_transform(image)

            # Get high frequencies
            high_frequencies = self.get_high_frequencies(fourier_transform)

            # Apply Inverse Fourier Transform to get the high-frequency image in the spatial domain
            high_freq_image = self.apply_inverse_fourier_transform(high_frequencies)
            high_freq_images.append(high_freq_image)

            # Optionally, save the high-frequency image (not saving in this example)
            #vutils.save_image(high_freq_image.cpu(), os.path.join(self.save_path, f"high_freq_{index}.png"))

        return torch.stack(high_freq_images)


class PatchEmbedding(nn.Module):
    """
    This class is used to extract patches from images and project them into embeddings.
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.linear_transform = nn.Linear(176 * 114, 768)
    def forward(self, x):
        x = self.proj(x).permute(0, 1, 3, 2)
        return x 

class cls_token_gen(nn.Module):
    """
    This class is used to generate the 512 dim token from 768 dim.
    """
    def __init__(self):
        super(cls_token_gen, self).__init__()
        # Define a linear layer to convert 768-dim to 512-dim
        self.linear = nn.Linear(768, 512)

    def forward(self, input):

        # Apply the linear layer to convert from 768 to 512 dimensions
        token = self.linear(input).squeeze(1)
        token = token.unsqueeze(1)
        return token

class kn_token_gen(nn.Module):
    """
    This class is used to generate the 512 dim token from 300 dim.
    """
    def __init__(self):
        super(kn_token_gen, self).__init__()
        # Define a linear layer to convert 768-dim to 512-dim
        self.linear = nn.Linear(300, 512)

    def forward(self, input):

        # Apply the linear layer to convert from 768 to 512 dimensions
        token = self.linear(input).squeeze(1)
        token = token.unsqueeze(1)
        return token
    
class content_prompt(nn.Module):
    """
    This class is used to generate the content prompt from the high-frequency features.
    """

    def __init__(self):
        super(content_prompt, self).__init__()

        # Convolutional layers to process the high-frequency features (hf_feats)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Linear layer to map the output of the convolutional layers to 512 dimensions
        self.fc = nn.Linear(256, 512)

    def forward(self, hf_feats):
        # Pass high-frequency features through the convolutional block
        x = self.conv_block(hf_feats)  # Output shape: (batch_size, 256, 1, 1)

        # Flatten the output to (batch_size, 256)
        x = x.view(x.size(0), -1)

        # Apply the linear layer to get the final output shape (batch_size, 512)
        x_out = self.fc(x)

        return x_out


class img_token_gen(nn.Module):
    """
    This class is used to generate the token from the patch embeddings.
    """
    def __init__(self):
        super(img_token_gen, self).__init__()
        self.lin1 = nn.Linear(768 * 14 * 14, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, 512)  # 768 (avg) -> 512

    def forward(self, patch_feats):
        # Reshape the patch features to (batch_size, 768, 14, 14)
        batch_size, _, h, w = patch_feats.shape
        
        # Apply adaptive average pooling
        avg_pooled_feats = self.global_avg_pool(patch_feats)  # (batch_size, 768, 1, 1)
        avg_pooled_feats = avg_pooled_feats.view(batch_size, -1)  # (batch_size, 768)
        
        # Transform to (batch_size, 512)
        output = self.fc(avg_pooled_feats)  # (batch_size, 512)
        
        output = output.unsqueeze(1)
        return output

        #196,768 AdaptiveAvgPool2d -> 768,1,1 -> 1,768-> 1,512