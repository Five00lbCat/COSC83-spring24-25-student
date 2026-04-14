import os
import random
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, hr_size=1024, lr_size=256, patch_size=96, scale_factors=[2, 3, 4], 
                 downsample_methods=['bicubic', 'bilinear', 'nearest'],
                 augment=True):
        """
        Dataset for superresolution with random downsampling
        
        Args:
            hr_dir (str): Directory containing high-resolution images
            hr_size (int): Size to resize high-resolution images (both width and height)
            lr_size (int): Fixed size for low-resolution images after downsampling
            patch_size (int): Size of HR patches to extract
            scale_factors (list): List of downsampling scale factors to randomly choose from
            downsample_methods (list): List of downsampling methods to randomly choose from
            augment (bool): Whether to use data augmentation
        """
        self.hr_dir = hr_dir
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.patch_size = patch_size
        self.scale_factors = scale_factors
        self.downsample_methods = downsample_methods
        self.augment = augment
        
        # Get file list
        self.image_files = []
        for f in os.listdir(hr_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path = os.path.join(hr_dir, f)
                if os.path.isfile(full_path):
                    self.image_files.append(f)
        
        print(f"Found {len(self.image_files)} image files in {hr_dir}")
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.image_files)
    
    # 10%
    def __getitem__(self, idx):
        """
        Get a training sample: low-resolution and high-resolution image pair

        Args:
            idx (int): Index of the image to load

        Returns:
            dict: Dictionary containing:
                - 'lr': Low-resolution image tensor  [3, patch//scale, patch//scale]
                - 'hr': High-resolution image tensor [3, patch_size,   patch_size  ]
                - 'scale_factor': Scale factor used for downsampling (torch.Tensor scalar)
                - 'method': Downsampling method used
                - 'filename': Original filename
                - 'error': Error message (only present if an error occurred)
        """
        try:
            filename = self.image_files[idx]
            img_path = os.path.join(self.hr_dir, filename)

            # Load and convert to RGB (handles grayscale / RGBA source images)
            hr_img = Image.open(img_path).convert('RGB')

            # Resize to a fixed square so all patches come from the same resolution
            hr_img = hr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC)

            # Random crop to patch_size × patch_size
            hr_patch = self._random_crop(hr_img)

            # Data augmentation (flips + 90° rotations)
            if self.augment:
                hr_patch = self._augment(hr_patch)

            # Pick a random scale factor and downsampling method for this sample.
            # NOTE: When using SuperResolutionDataset (multiple scale_factors), different
            # items may produce LR tensors with different spatial sizes, which will cause
            # PyTorch's default collation to fail. Use FixedScaleDataset (or a custom
            # collate_fn) for multi-scale training.
            scale_factor = random.choice(self.scale_factors)
            method = random.choice(self.downsample_methods)

            # Downsample HR patch → LR patch  (shape: patch_size//scale × patch_size//scale)
            lr_patch = self._downsample(hr_patch, scale_factor, method)

            # ToTensor normalises uint8 [0, 255] → float32 [0, 1]
            hr_tensor = self.to_tensor(hr_patch)
            lr_tensor = self.to_tensor(lr_patch)

            return {
                'lr': lr_tensor,
                'hr': hr_tensor,
                'scale_factor': torch.tensor(scale_factor),
                'method': method,
                'filename': filename,
            }

        except Exception as e:
            print(f"Error loading index {idx} "
                  f"({self.image_files[idx] if idx < len(self.image_files) else 'unknown'}): {e}")
            # Return zero tensors with a consistent shape so the DataLoader does not crash
            sf = self.scale_factors[0]
            lr_side = self.patch_size // sf
            return {
                'lr': torch.zeros(3, lr_side, lr_side),
                'hr': torch.zeros(3, self.patch_size, self.patch_size),
                'scale_factor': torch.tensor(sf),
                'method': self.downsample_methods[0],
                'filename': self.image_files[idx] if idx < len(self.image_files) else 'unknown',
                'error': str(e),
            }
    
    def _random_crop(self, img):
        """Safely crop a patch from the image"""
        width, height = img.size
        
        # Ensure the image is large enough for cropping
        if width < self.patch_size or height < self.patch_size:
            # Resize if too small
            ratio = max(self.patch_size / width, self.patch_size / height) + 0.1  # Add small buffer
            new_width = max(int(width * ratio), self.patch_size)
            new_height = max(int(height * ratio), self.patch_size)
            try:
                img = img.resize((new_width, new_height), Image.BICUBIC)
                width, height = img.size
            except Exception as e:
                print(f"Resize error: {e}. Using original size and center crop.")
                # If resize fails, use center crop instead of random crop
                left = max(0, (width - self.patch_size) // 2)
                top = max(0, (height - self.patch_size) // 2)
                right = min(width, left + self.patch_size)
                bottom = min(height, top + self.patch_size)
                return img.crop((left, top, right, bottom))
        
        # Safe random crop
        max_left = max(0, width - self.patch_size)
        max_top = max(0, height - self.patch_size)
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        right = min(width, left + self.patch_size)
        bottom = min(height, top + self.patch_size)
        
        return img.crop((left, top, right, bottom))
    
    def _augment(self, img):
        """Apply random augmentations"""
        try:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                
            # Random vertical flip
            if random.random() > 0.5:
                img = TF.vflip(img)
                
            # Random 90-degree rotation (safer than arbitrary angles)
            if random.random() > 0.5:
                angle = random.choice([0, 90, 180, 270])
                img = img.rotate(angle, expand=False)
                
            return img
        except Exception as e:
            print(f"Augmentation error: {e}. Using original image.")
            return img
    
    def _downsample(self, hr_img, scale_factor, method):
        """Safely downsample image by scale factor using specified method"""
        try:
            width, height = hr_img.size
            lr_width = max(1, width // scale_factor)
            lr_height = max(1, height // scale_factor)
            
            # Convert method string to PIL resample mode
            resample_mode = {
                'bicubic': Image.BICUBIC,
                'bilinear': Image.BILINEAR,
                'nearest': Image.NEAREST,
                'lanczos': Image.LANCZOS
            }.get(method, Image.BICUBIC)
            
            # Downsample
            lr_img = hr_img.resize((lr_width, lr_height), resample_mode)
            
            # Optional: Add noise to simulate real-world low-res images
            if random.random() > 0.7:
                lr_img = self._add_noise(lr_img)
                
            return lr_img
        except Exception as e:
            print(f"Downsampling error: {e}. Using simpler method.")
            # Fallback to simpler method
            width, height = hr_img.size
            lr_width = max(1, width // scale_factor)
            lr_height = max(1, height // scale_factor)
            return hr_img.resize((lr_width, lr_height), Image.NEAREST)
    
    def _add_noise(self, img):
        """Add random noise to simulate real-world low-res images"""
        try:
            img_np = np.array(img).astype(np.float32)
            
            # Simple Gaussian noise (avoid complex noise types that might cause issues)
            noise_level = random.uniform(1, 10)
            noise = np.random.normal(0, noise_level, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
                
            return Image.fromarray(img_np.astype(np.uint8))
        except Exception as e:
            print(f"Noise addition error: {e}. Using original image.")
            return img


class FixedScaleDataset(SuperResolutionDataset):
    """A version of the dataset that uses a fixed scale factor for all images but random methods"""
    def __init__(self, hr_dir, scale_factor=4, hr_size=1024, lr_size=None,
                 patch_size=96, downsample_methods=['bicubic', 'bilinear', 'nearest', 'lanczos'], augment=True):
        """
        Dataset for superresolution with fixed downsampling scale but random methods
        
        Args:
            hr_dir (str): Directory containing high-resolution images
            scale_factor (int): Fixed scale factor to use for all images
            hr_size (int): Size to resize high-resolution images (both width and height)
            lr_size (int): Size of LR images (if None, will be hr_size // scale_factor)
            patch_size (int): Size of HR patches to extract
            downsample_methods (list): List of methods to randomly choose from for downsampling
            augment (bool): Whether to use data augmentation
        """
        # Calculate LR size if not provided
        if lr_size is None:
            lr_size = hr_size // scale_factor
            
        super().__init__(
            hr_dir=hr_dir, 
            hr_size=hr_size, 
            lr_size=lr_size,
            patch_size=patch_size, 
            scale_factors=[scale_factor], 
            downsample_methods=downsample_methods,
            augment=augment
        )


def get_dataloader(hr_dir, batch_size=16, patch_size=96, num_workers=4, 
                  scale_factors=None, fixed_scale=None, 
                  downsample_methods=['bicubic', 'bilinear', 'nearest', 'lanczos']):
    """Create dataloader for superresolution training
    
    Args:
        hr_dir (str): Directory containing high-resolution images
        batch_size (int): Batch size
        patch_size (int): Size of HR patches to extract
        num_workers (int): Number of worker processes
        scale_factors (list): List of scale factors for random downsampling
        fixed_scale (int): If set, use a fixed scale factor for all images
        downsample_methods (list): List of downsampling methods to randomly choose from
    """
    if fixed_scale is not None:
        # Use fixed scale dataset
        dataset = FixedScaleDataset(
            hr_dir=hr_dir,
            scale_factor=fixed_scale,
            hr_size=1024,
            patch_size=patch_size,
            downsample_methods=downsample_methods,
            augment=True
        )
    else:
        # Use random scale dataset
        dataset = SuperResolutionDataset(
            hr_dir=hr_dir,
            hr_size=1024,
            patch_size=patch_size,
            scale_factors=scale_factors or [2, 3, 4],
            downsample_methods=downsample_methods,
            augment=True
        )
    
    # Safer DataLoader configuration
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False if num_workers == 0 else True,
    )
    
    return dataloader


# Visualization utility function
def visualize_batch(batch, max_samples=4):
    """
    Visualize a batch of superresolution data
    
    Args:
        batch (dict): Batch from dataloader
        max_samples (int): Maximum number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    lr_imgs = batch['lr']
    hr_imgs = batch['hr']
    scale_factors = batch['scale_factor']
    methods = batch['method']
    
    # Limit to max_samples
    n_samples = min(max_samples, lr_imgs.size(0))
    
    # Create a figure with n_samples rows and 2 columns
    plt.figure(figsize=(10, 4 * n_samples))
    
    for i in range(n_samples):
        # Convert tensors to numpy for visualization
        lr_img = lr_imgs[i].permute(1, 2, 0).numpy()
        hr_img = hr_imgs[i].permute(1, 2, 0).numpy()
        
        # Clip to valid range [0, 1]
        lr_img = np.clip(lr_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        # Add LR image
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.imshow(lr_img)
        plt.title(f"LR (Scale: {scale_factors[i].item()}x, Method: {methods[i]})")
        plt.axis('off')
        
        # Add HR image
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.imshow(hr_img)
        plt.title("HR")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example directory path - update to your DIV2K path
    hr_dir = "DIV2K_train_HR"
    
    # Fixed scale with random downsampling methods
    dataloader = get_dataloader(
        hr_dir=hr_dir,
        batch_size=8,
        patch_size=128,
        fixed_scale=4,  # Use fixed 4x downsampling for all images
        downsample_methods=['bicubic', 'bilinear', 'nearest', 'lanczos'],
        num_workers=0  # Use single-process loading for debugging
    )
    
    print("Testing dataloader...")
    try:
        # Test loading a few batches
        for i, batch in enumerate(dataloader):
            if i == 0:
                lr_imgs = batch['lr']
                hr_imgs = batch['hr']
                scale_factors = batch['scale_factor']
                methods = batch['method']
                
                print(f"LR shape: {lr_imgs.shape}, HR shape: {hr_imgs.shape}")
                print(f"Scale factors in batch: {scale_factors}")
                print(f"Downsampling methods: {methods}")
                
                # Visualize the first batch
                visualize_batch(batch)
            
            if i >= 2:  # Only test a few batches
                break
                
        print("Dataloader test successful!")
    except Exception as e:
        print(f"Dataloader test failed: {str(e)}")