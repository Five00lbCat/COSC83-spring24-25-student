"""
Evaluate a trained SuperResolutionCNN against the bicubic baseline.

For each image in --img_dir:
  1. Load full-resolution image as HR ground truth.
  2. Downsample by --scale_factor (default 2) with bicubic to produce LR.
  3. Upsample LR with bicubic → bicubic baseline.
  4. Run LR through the trained model → SR output.
  5. Compute PSNR and SSIM for both SR and bicubic vs. HR.
  6. Save a side-by-side figure: LR | Bicubic | SR | HR.
Print a per-image table and an aggregate summary at the end.

Usage:
    python evaluate.py \\
        --checkpoint checkpoints/best_model.pth \\
        --img_dir    ../data/DIV2K/DIV2K_valid_HR \\
        --out_dir    eval_results \\
        --scale_factor 2
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')   # headless-safe; no display required
import matplotlib.pyplot as plt
from PIL import Image

# ── allow running from any directory ─────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from srcnn   import SuperResolutionCNN
from metrics import calculate_psnr, calculate_ssim


# ── helpers ───────────────────────────────────────────────────────────────────

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → float32 tensor [0,1] shape [3,H,W]."""
    return T.ToTensor()(img)


def to_pil(t: torch.Tensor) -> Image.Image:
    """Float [0,1] tensor [3,H,W] → PIL."""
    return TF.to_pil_image(t.clamp(0, 1).cpu())


def bicubic_downsample(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """Downsample [3,H,W] by integer scale using bicubic, return [3,H/s,W/s]."""
    h, w = hr.shape[1], hr.shape[2]
    return F.interpolate(
        hr.unsqueeze(0),
        size=(h // scale, w // scale),
        mode='bicubic',
        align_corners=False,
    ).squeeze(0).clamp(0, 1)


def bicubic_upsample(lr: torch.Tensor, hr_size: tuple) -> torch.Tensor:
    """Upsample [3,h,w] to hr_size (H,W) using bicubic."""
    return F.interpolate(
        lr.unsqueeze(0),
        size=hr_size,
        mode='bicubic',
        align_corners=False,
    ).squeeze(0).clamp(0, 1)


def make_comparison_figure(
    lr: torch.Tensor,
    bicubic: torch.Tensor,
    sr: torch.Tensor,
    hr: torch.Tensor,
    filename: str,
    bic_psnr: float,
    bic_ssim: float,
    sr_psnr: float,
    sr_ssim: float,
) -> plt.Figure:
    """
    Four-panel figure: LR (bicubic-upscaled for display) | Bicubic | SR | HR.
    LR is shown upscaled to HR resolution so all panels share the same pixel
    dimensions in the figure — makes alignment artifacts easier to spot.
    """
    hr_size = (hr.shape[1], hr.shape[2])
    lr_display = bicubic_upsample(lr, hr_size)   # only for display

    panels = [
        (lr_display, f'LR input\n(×{hr_size[0]//lr.shape[1]} upscaled for display)'),
        (bicubic,    f'Bicubic\nPSNR {bic_psnr:.2f} dB  SSIM {bic_ssim:.4f}'),
        (sr,         f'SR model\nPSNR {sr_psnr:.2f} dB  SSIM {sr_ssim:.4f}'),
        (hr,         'HR ground truth'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(filename, fontsize=11, y=1.01)

    for ax, (img_t, title) in zip(axes, panels):
        ax.imshow(to_pil(img_t))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--checkpoint',   required=True,
                   help='Path to trained checkpoint (.pth)')
    p.add_argument('--img_dir',      required=True,
                   help='Directory of high-resolution evaluation images')
    p.add_argument('--out_dir',      default='eval_results',
                   help='Directory to save comparison figures (default: eval_results)')
    p.add_argument('--scale_factor', type=int, default=2,
                   help='Downsampling scale factor used during training (default: 2)')
    p.add_argument('--num_features', type=int, default=64)
    p.add_argument('--num_blocks',   type=int, default=16)
    p.add_argument('--max_images',   type=int, default=None,
                   help='Cap the number of images evaluated (useful for quick checks)')
    p.add_argument('--max_size',     type=int, default=None,
                   help='Resize images so the long edge is at most this many pixels '
                        'before evaluation. Recommended for large photos (e.g. 1024).')
    p.add_argument('--no_figures',   action='store_true',
                   help='Skip saving per-image figures (faster for large sets)')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device        : {device}')

    # ── load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f'Checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location=device)
    model = SuperResolutionCNN(
        scale_factor=args.scale_factor,
        num_channels=3,
        num_features=args.num_features,
        num_blocks=args.num_blocks,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    trained_epoch = ckpt.get('epoch', '?')
    trained_psnr  = ckpt.get('psnr',  float('nan'))
    print(f'Checkpoint    : {ckpt_path.name}')
    print(f'Trained epoch : {trained_epoch}   best val PSNR : {trained_psnr:.2f} dB')
    print(f'Scale factor  : {args.scale_factor}×')

    # ── find images ───────────────────────────────────────────────────────────
    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        sys.exit(f'Image directory not found: {img_dir}')

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        sys.exit(f'No images found in {img_dir}')

    if args.max_images is not None:
        images = images[:args.max_images]

    print(f'Images        : {len(images)}')
    if args.max_size:
        print(f'Max size      : {args.max_size}px (long edge)')

    out_dir = Path(args.out_dir)
    if not args.no_figures:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-image evaluation ──────────────────────────────────────────────────
    col_w = max(len(p.name) for p in images)

    header = (f"{'Image':<{col_w}}  {'W×H':>10}  "
              f"{'Bic PSNR':>9}  {'Bic SSIM':>9}  "
              f"{'SR PSNR':>8}  {'SR SSIM':>8}  "
              f"{'ΔPSNR':>7}  {'ΔSSIM':>7}")
    sep = '-' * len(header)
    print(f'\n{sep}\n{header}\n{sep}')

    records = []

    for img_path in images:
        try:
            hr_pil = load_image(img_path)
        except Exception as e:
            print(f'  SKIP {img_path.name}: {e}')
            continue

        # Optionally downscale very large images to avoid OOM on CPU
        if args.max_size:
            w_orig, h_orig = hr_pil.size
            long_edge = max(w_orig, h_orig)
            if long_edge > args.max_size:
                scale = args.max_size / long_edge
                hr_pil = hr_pil.resize(
                    (int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS
                )

        hr_t = to_tensor(hr_pil)                              # [3,H,W]
        H, W = hr_t.shape[1], hr_t.shape[2]

        # Crop to dimensions divisible by scale so LR→SR is exact integer mult.
        # Without this, F.interpolate will produce sizes that are off by 1.
        H_crop = (H // args.scale_factor) * args.scale_factor
        W_crop = (W // args.scale_factor) * args.scale_factor
        hr_t = hr_t[:, :H_crop, :W_crop]

        lr_t      = bicubic_downsample(hr_t, args.scale_factor)
        bicubic_t = bicubic_upsample(lr_t, (H_crop, W_crop))

        # Model inference — add / remove batch dim
        with torch.no_grad():
            sr_t = model(lr_t.unsqueeze(0).to(device)).squeeze(0).clamp(0, 1).cpu()

        # PSNR / SSIM expect [C,H,W] single-image tensors
        bic_psnr = calculate_psnr(bicubic_t, hr_t)
        bic_ssim = calculate_ssim(bicubic_t, hr_t)
        sr_psnr  = calculate_psnr(sr_t, hr_t)
        sr_ssim  = calculate_ssim(sr_t, hr_t)

        records.append({
            'name':     img_path.name,
            'w': W_crop, 'h': H_crop,
            'bic_psnr': bic_psnr, 'bic_ssim': bic_ssim,
            'sr_psnr':  sr_psnr,  'sr_ssim':  sr_ssim,
        })

        d_psnr = sr_psnr - bic_psnr
        d_ssim = sr_ssim - bic_ssim
        sign_p = '+' if d_psnr >= 0 else ''
        sign_s = '+' if d_ssim >= 0 else ''

        print(
            f'{img_path.name:<{col_w}}  {W_crop}×{H_crop:>{9-len(str(W_crop))}}  '
            f'{bic_psnr:>9.2f}  {bic_ssim:>9.4f}  '
            f'{sr_psnr:>8.2f}  {sr_ssim:>8.4f}  '
            f'{sign_p}{d_psnr:>6.2f}  {sign_s}{d_ssim:>6.4f}'
        )

        if not args.no_figures:
            fig = make_comparison_figure(
                lr_t, bicubic_t, sr_t, hr_t,
                img_path.name,
                bic_psnr, bic_ssim, sr_psnr, sr_ssim,
            )
            fig_path = out_dir / (img_path.stem + '_comparison.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    if not records:
        print('No images were successfully processed.')
        return

    # ── summary ───────────────────────────────────────────────────────────────
    n = len(records)
    avg = lambda key: sum(r[key] for r in records) / n

    avg_bic_psnr = avg('bic_psnr')
    avg_bic_ssim = avg('bic_ssim')
    avg_sr_psnr  = avg('sr_psnr')
    avg_sr_ssim  = avg('sr_ssim')

    print(sep)
    print(
        f'{"AVERAGE":<{col_w}}  {"":>10}  '
        f'{avg_bic_psnr:>9.2f}  {avg_bic_ssim:>9.4f}  '
        f'{avg_sr_psnr:>8.2f}  {avg_sr_ssim:>8.4f}  '
        f'{avg_sr_psnr - avg_bic_psnr:>+7.2f}  {avg_sr_ssim - avg_bic_ssim:>+7.4f}'
    )
    print(sep)

    print(f'\nSummary over {n} image(s):')
    print(f'  Bicubic  — PSNR: {avg_bic_psnr:.2f} dB   SSIM: {avg_bic_ssim:.4f}')
    print(f'  SR model — PSNR: {avg_sr_psnr:.2f} dB   SSIM: {avg_sr_ssim:.4f}')
    psnr_delta = avg_sr_psnr - avg_bic_psnr
    ssim_delta = avg_sr_ssim - avg_bic_ssim
    sign = '+' if psnr_delta >= 0 else ''
    print(f'  Gain     — PSNR: {sign}{psnr_delta:.2f} dB   SSIM: {sign}{ssim_delta:.4f}')

    if not args.no_figures:
        print(f'\nFigures saved to: {out_dir.resolve()}/')


if __name__ == '__main__':
    main()
