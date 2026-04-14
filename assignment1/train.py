import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Import your dataloader and model
from dataloader import get_dataloader
from srcnn import SuperResolutionCNN

# Metrics
from metrics import calculate_psnr, calculate_ssim, fast_psnr, fast_ssim


# ── Bonus option B: VGG perceptual loss ─────────────────────────────────────
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 relu3_3 features (Wang et al., ECCV 2018 style).

    Both SR and HR images are expected in [0, 1] — we re-normalise to
    ImageNet mean/std before passing to VGG to match the network's training
    distribution, which produces more meaningful feature distances.

    Non-obvious: we stop at relu3_3 (index 16 in vgg.features) rather than
    deeper layers. Deeper features encode semantics; shallower features
    encode texture and edges — which is what we care about for SR.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        # relu3_3 is the output of the 16th child (0-indexed) of vgg.features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).to(device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants (kept as buffers so .to(device) works)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406],
                                                   device=device).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225],
                                                   device=device).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        return F.l1_loss(self.feature_extractor(sr_norm),
                         self.feature_extractor(hr_norm))


#10%
def train(config):
    """
    Train the SuperResolution model

    Args:
        config (dict): Configuration parameters — see __main__ block for keys.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['sample_dir'], exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SuperResolutionCNN(
        scale_factor=config['scale_factor'],
        num_channels=3,
        num_features=config['num_features'],
        num_blocks=config['num_blocks'],
    ).to(device)

    start_epoch = 0
    if config['resume'] and os.path.exists(config['resume']):
        checkpoint = torch.load(config['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    # ── Loss functions ───────────────────────────────────────────────────────
    criterion = nn.L1Loss()

    # Bonus option B: perceptual loss on top of L1
    use_perceptual = config.get('use_perceptual_loss', False)
    perceptual_loss_fn = VGGPerceptualLoss(device) if use_perceptual else None
    perceptual_weight  = config.get('perceptual_loss_weight', 0.1)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=config['lr_decay_step'],
                                          gamma=config['lr_decay_gamma'])

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataloader = get_dataloader(
        hr_dir=config['train_dir'],
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        fixed_scale=config['scale_factor'],
        downsample_methods=config['downsample_methods'],
        num_workers=config['num_workers'],
    )
    val_dataloader = get_dataloader(
        hr_dir=config['val_dir'],
        batch_size=config['batch_size'],
        patch_size=config['patch_size'],
        fixed_scale=config['scale_factor'],
        downsample_methods=['bicubic'],   # deterministic validation
        num_workers=config['num_workers'],
    )

    # ── History ──────────────────────────────────────────────────────────────
    train_losses = []
    val_losses   = []
    val_psnrs    = []
    val_ssims    = []
    best_psnr    = 0.0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config['num_epochs']):
        # ── Train phase ──────────────────────────────────────────────────────
        model.train()
        epoch_loss   = 0.0
        n_train_batches = 0

        pbar = tqdm(train_dataloader,
                    desc=f"Epoch {epoch+1}/{config['num_epochs']} [train]",
                    leave=False)
        for batch in pbar:
            lr_imgs = batch['lr'].to(device)   # [B, 3, H/s, W/s]
            hr_imgs = batch['hr'].to(device)   # [B, 3, H,   W  ]

            optimizer.zero_grad()

            sr_imgs = model(lr_imgs)           # [B, 3, H,   W  ]

            loss = criterion(sr_imgs, hr_imgs)

            # Perceptual loss (bonus B): clamp SR to [0,1] before VGG to
            # avoid out-of-range activations destabilising feature stats.
            if use_perceptual and perceptual_loss_fn is not None:
                p_loss = perceptual_loss_fn(sr_imgs.clamp(0, 1), hr_imgs)
                loss   = loss + perceptual_weight * p_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = epoch_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)

        # Placeholder so val_* lists stay aligned with train_losses
        val_losses.append(None)
        val_psnrs.append(None)
        val_ssims.append(None)

        scheduler.step()

        # ── Validation phase ─────────────────────────────────────────────────
        do_val = ((epoch + 1) % config['validation_interval'] == 0
                  or epoch == config['num_epochs'] - 1)

        if do_val:
            model.eval()
            v_loss = v_psnr = v_ssim = bicubic_psnr = 0.0
            n_val = 0

            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    if i >= config['val_batch_limit']:
                        break

                    lr_imgs = batch['lr'].to(device)
                    hr_imgs = batch['hr'].to(device)

                    sr_imgs      = model(lr_imgs)
                    sr_clamped   = sr_imgs.clamp(0, 1)

                    v_loss += criterion(sr_imgs, hr_imgs).item()
                    v_psnr += fast_psnr(sr_clamped, hr_imgs)
                    v_ssim += fast_ssim(sr_clamped, hr_imgs)

                    # Bicubic baseline: upsample LR to HR spatial size
                    # Non-obvious: align_corners=False is the correct setting for
                    # image upscaling (aligns pixel centres, not corners).
                    bicubic_up = F.interpolate(
                        lr_imgs,
                        size=hr_imgs.shape[2:],
                        mode='bicubic',
                        align_corners=False,
                    ).clamp(0, 1)
                    bicubic_psnr += fast_psnr(bicubic_up, hr_imgs)

                    # Save a sample grid from the first validation batch
                    if i == 0:
                        _save_samples(
                            lr_imgs, sr_clamped, hr_imgs,
                            bicubic_up=bicubic_up,
                            path=os.path.join(
                                config['sample_dir'],
                                f'epoch_{epoch+1:04d}.png'
                            ),
                        )

                    n_val += 1

            avg_v_loss    = v_loss    / max(n_val, 1)
            avg_v_psnr    = v_psnr    / max(n_val, 1)
            avg_v_ssim    = v_ssim    / max(n_val, 1)
            avg_bic_psnr  = bicubic_psnr / max(n_val, 1)

            # Backfill the placeholder appended above
            val_losses[-1] = avg_v_loss
            val_psnrs[-1]  = avg_v_psnr
            val_ssims[-1]  = avg_v_ssim

            print(
                f"Epoch {epoch+1:4d} | train_loss={avg_train_loss:.4f} "
                f"| val_loss={avg_v_loss:.4f} "
                f"| PSNR={avg_v_psnr:.2f} dB  (bicubic: {avg_bic_psnr:.2f} dB) "
                f"| SSIM={avg_v_ssim:.4f}"
            )

            # Save best model
            if avg_v_psnr > best_psnr:
                best_psnr = avg_v_psnr
                torch.save(
                    {'epoch': epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'psnr': best_psnr},
                    os.path.join(config['checkpoint_dir'], 'best_model.pth'),
                )
                print(f"  → new best PSNR {best_psnr:.2f} dB — checkpoint saved")
        else:
            print(f"Epoch {epoch+1:4d} | train_loss={avg_train_loss:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save(
                {'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict()},
                os.path.join(config['checkpoint_dir'],
                             f'checkpoint_epoch_{epoch+1:04d}.pth'),
            )

    # ── Plot training history ────────────────────────────────────────────────
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    val_epochs      = [i for i, v in enumerate(val_losses) if v is not None]
    val_loss_values = [v for v in val_losses if v is not None]
    plt.plot(val_epochs, val_loss_values, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 3, 2)
    val_psnr_values = [v for v in val_psnrs if v is not None]
    plt.plot(val_epochs, val_psnr_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR on Validation Set')

    plt.subplot(1, 3, 3)
    val_ssim_values = [v for v in val_ssims if v is not None]
    plt.plot(val_epochs, val_ssim_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM on Validation Set')

    plt.tight_layout()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_history.png'))
    print("Training completed!")


def _save_samples(lr_imgs, sr_imgs, hr_imgs, bicubic_up, path, n=4):
    """
    Save a side-by-side grid: bicubic-upsampled LR | SR | HR.
    The bicubic_up tensor (same size as SR/HR) is used for the LR column
    so all three panels share the same spatial dimensions in the grid.
    """
    n = min(n, lr_imgs.size(0))
    # Stack as [bicubic | SR | HR] per row
    rows = []
    for i in range(n):
        rows.append(torch.cat([bicubic_up[i], sr_imgs[i], hr_imgs[i]], dim=2))
    grid = torch.stack(rows, dim=0)           # [n, 3, H, 3W]
    save_image(grid, path, nrow=1)


if __name__ == "__main__":
    config = {
        # Model parameters
        'scale_factor':   4,
        'num_features':   64,
        'num_blocks':     16,

        # Data parameters
        'train_dir':            'DIV2K_train_HR',
        'val_dir':              'DIV2K_valid_HR',
        'patch_size':           128,
        'downsample_methods':   ['bicubic', 'bilinear', 'nearest', 'lanczos'],

        # Training parameters
        'batch_size':           16,
        'num_epochs':           10,
        'learning_rate':        1e-4,
        'lr_decay_step':        30,
        'lr_decay_gamma':       0.5,
        'num_workers':          4,
        'validation_interval':  1,
        'val_batch_limit':      10,

        # Bonus option B — set True to add VGG perceptual loss
        'use_perceptual_loss':   False,
        'perceptual_loss_weight': 0.1,

        # Checkpoint parameters
        'checkpoint_dir': 'checkpoints',
        'sample_dir':     'samples',
        'save_every':     5,
        'resume':         None,
    }

    train(config)
