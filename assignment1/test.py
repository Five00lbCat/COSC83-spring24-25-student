import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

# Import your model and metrics
from srcnn import SuperResolutionCNN
from metrics import calculate_psnr, calculate_ssim

def test_single_image(model, lr_img, scale_factor, device):
    """
    Test model on a single image
    
    Args:
        model: The trained superresolution model
        lr_img: Low resolution input image (PIL Image)
        scale_factor: Upscaling factor
        device: Device to run inference on
    
    Returns:
        Super-resolution output as PIL Image
    """
    # Convert to tensor
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = sr_tensor.clamp(0.0, 1.0)
    
    # Convert back to PIL Image
    to_pil = transforms.ToPILImage()
    sr_img = to_pil(sr_tensor.squeeze(0).cpu())
    
    return sr_img

def test_model(config):
    """
    Test the trained model on test images
    
    Args:
        config (dict): Configuration parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create the model
    model = SuperResolutionCNN(
        scale_factor=config['scale_factor'],
        num_channels=3,
        num_features=config['num_features'],
        num_blocks=config['num_blocks']
    ).to(device)
    
    # Load the trained model
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {config['model_path']}")
    print(f"Model was trained for {checkpoint['epoch']} epochs")
    print(f"Best PSNR: {checkpoint.get('psnr', float('nan')):.2f}, SSIM: {checkpoint.get('ssim', float('nan')):.4f}")
    
    # Get all test images
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        test_images.extend(glob.glob(os.path.join(config['test_dir'], ext)))
    
    print(f"Found {len(test_images)} test images")
    
    # Initialize metrics lists
    psnr_values = []
    ssim_values = []
    
    # Process each test image
    for img_path in tqdm(test_images):
        # Load HR image
        hr_img = Image.open(img_path).convert('RGB')
        hr_img = hr_img.resize((1024, 1024), Image.BICUBIC)
        # Create LR image by downsampling
        width, height = hr_img.size
        lr_width = width // config['scale_factor']
        lr_height = height // config['scale_factor']
        lr_img = hr_img.resize((lr_width, lr_height), Image.BICUBIC)
        
        # Process with model
        sr_img = test_single_image(model, lr_img, config['scale_factor'], device)
        
        
        # Calculate metrics
        if config['calculate_metrics']:
            # Resize HR image to match SR output if sizes don't match
            if hr_img.size != sr_img.size:
                hr_img = hr_img.resize(sr_img.size, Image.BICUBIC)
            
            # Convert to tensors
            to_tensor = transforms.ToTensor()
            hr_tensor = to_tensor(hr_img)
            sr_tensor = to_tensor(sr_img)
            
            # Calculate PSNR and SSIM
            psnr = calculate_psnr(sr_tensor, hr_tensor)
            ssim = calculate_ssim(sr_tensor, hr_tensor)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        
        # Save output images
        if config['save_images']:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Resize HR image to match SR output if sizes don't match
            if hr_img.size != sr_img.size:
                hr_img = hr_img.resize(sr_img.size, Image.BICUBIC)
            
            # Save individual images
            lr_img.save(os.path.join(config['output_dir'], f"{filename}_LR.png"))
            sr_img.save(os.path.join(config['output_dir'], f"{filename}_SR.png"))
            hr_img.save(os.path.join(config['output_dir'], f"{filename}_HR.png"))
            
            # Save comparison image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(np.array(lr_img))
            axes[0].set_title("Low Resolution")
            axes[0].axis('off')
            
            axes[1].imshow(np.array(sr_img))
            axes[1].set_title("Super Resolution")
            axes[1].axis('off')
            
            axes[2].imshow(np.array(hr_img))
            axes[2].set_title("High Resolution (Ground Truth)")
            axes[2].axis('off')
            
            if config['calculate_metrics']:
                plt.suptitle(f"PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(config['output_dir'], f"{filename}_comparison.png"))
            plt.close()
    
    # Calculate average metrics
    if config['calculate_metrics']:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print(f"Average PSNR: {avg_psnr:.2f}dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Save metrics to file
    if config['calculate_metrics']:
        with open(os.path.join(config['output_dir'], 'metrics.txt'), 'w') as f:
            f.write(f"Model: {config['model_path']}\n")
            f.write(f"Scale factor: {config['scale_factor']}\n")
            f.write(f"Number of test images: {len(test_images)}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}dB\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Super Resolution model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output images")
    parser.add_argument("--scale_factor", type=int, default=4, help="Super resolution scale factor")
    parser.add_argument("--num_features", type=int, default=64, help="Number of feature channels")
    parser.add_argument("--num_blocks", type=int, default=16, help="Number of residual blocks")
    parser.add_argument("--save_images", action="store_true", help="Save output images")
    parser.add_argument("--calculate_metrics", action="store_true", help="Calculate PSNR and SSIM metrics")
    
    args = parser.parse_args()
    
    config = {
        'model_path': args.model_path,
        'test_dir': args.test_dir,
        'output_dir': args.output_dir,
        'scale_factor': args.scale_factor,
        'num_features': args.num_features,
        'num_blocks': args.num_blocks,
        'save_images': args.save_images,
        'calculate_metrics': args.calculate_metrics,
    }
    
    test_model(config)

    #python test.py --model_path checkpoints/best_model.pth --test_dir /Users/f007fxx/Desktop/COSC83-spring24-25/assignment1/DIV2K_valid_HR --output_dir results --save_images --calculate_metrics