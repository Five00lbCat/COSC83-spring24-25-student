import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#5%
class ResidualBlock(nn.Module):
    """
    Residual block: Conv→BN→ReLU→Conv→BN with an identity skip connection,
    followed by a final ReLU.

    Pattern follows He et al. (CVPR 2016) post-activation style:
        out = ReLU(BN(conv2(ReLU(BN(conv1(x))))) + x)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Skip connection adds before the final activation
        return self.relu(out + residual)


#5%
class UpscaleBlock(nn.Module):
    """
    Sub-pixel convolution (PixelShuffle) upscaling block.

    Strategy (Shi et al., CVPR 2016):
        1. Expand channels from C → C * scale² with a convolution.
        2. PixelShuffle rearranges [B, C*s², H, W] → [B, C, H*s, W*s].
        3. ReLU activation.

    This avoids the checkerboard artefacts that transposed convolutions can produce
    because every output pixel is generated from distinct input features.
    """
    def __init__(self, in_channels: int, scale_factor: int):
        super().__init__()
        # The conv must output in_channels * scale_factor² channels so that after
        # PixelShuffle the spatial dims are expanded while channel count returns to in_channels.
        out_channels = in_channels * (scale_factor ** 2)
        self.conv         = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu          = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pixel_shuffle(self.conv(x)))


#10%
class SuperResolutionCNN(nn.Module):
    """
    Super-Resolution CNN with:
      - 9×9 initial feature extraction conv
      - N residual blocks (default 16)
      - 3×3 mid conv + BN (anchors the global skip)
      - Global skip connection across all residual blocks
      - PixelShuffle upscaling (×2 blocks for powers-of-2 scales; ×3 for scale=3)
      - 9×9 reconstruction conv

    Supports scale factors: 2, 3, 4 (and 8 by extension of the ×2 chain).
    """
    def __init__(self, scale_factor: int = 4, num_channels: int = 3,
                 num_features: int = 64, num_blocks: int = 16):
        super().__init__()
        self.scale_factor = scale_factor

        # ── Initial feature extraction ──────────────────────────────────────
        # Large 9×9 kernel captures wide receptive field at the start.
        # padding=4 preserves spatial size.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )

        # ── Residual body ────────────────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # ── Mid conv after residual stack ────────────────────────────────────
        # No activation here — the global skip is added immediately after,
        # and we don't want to cut off negative values before that addition.
        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
        )

        # ── Upscaling chain ──────────────────────────────────────────────────
        # Powers of 2: chain ×2 blocks (log₂ steps). Scale=3: single ×3 block.
        # We never combine ×2 and ×3 blocks; unsupported scales raise immediately.
        if scale_factor in (2, 4, 8):
            n_steps = int(math.log2(scale_factor))   # 1, 2, or 3 ×2 blocks
            self.upscale = nn.Sequential(
                *[UpscaleBlock(num_features, 2) for _ in range(n_steps)]
            )
        elif scale_factor == 3:
            self.upscale = nn.Sequential(UpscaleBlock(num_features, 3))
        else:
            raise ValueError(
                f"Unsupported scale_factor={scale_factor}. "
                "Use 2, 3, 4, or 8."
            )

        # ── Final reconstruction ─────────────────────────────────────────────
        # Another wide 9×9 kernel; no activation — output should be in [0,1]
        # after clamping, not forced through ReLU which would clip negatives.
        self.final_conv = nn.Conv2d(num_features, num_channels, kernel_size=9, padding=4)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming normal init for conv weights (fan_in mode, compatible with ReLU);
        standard BN init (weight=1, bias=0).
        Conv biases are zeroed.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: initial feature extraction → store for global skip
        initial_features = self.initial_conv(x)         # [B, F, H, W]

        # Step 2: residual blocks
        features = self.res_blocks(initial_features)    # [B, F, H, W]

        # Step 3: mid conv
        features = self.mid_conv(features)              # [B, F, H, W]

        # Step 4: global skip — adds initial features back, preventing
        # the deep residual stack from diverging (same idea as ResNet's stem skip)
        features = features + initial_features          # [B, F, H, W]

        # Step 5: upscaling
        features = self.upscale(features)               # [B, F, H*s, W*s]

        # Step 6: final reconstruction conv
        out = self.final_conv(features)                 # [B, C, H*s, W*s]

        return out
