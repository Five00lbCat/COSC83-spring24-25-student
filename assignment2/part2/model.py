"""
Model architecture for Siamese Neural Network
"""
import torch
import torch.nn as nn
import torchvision.models as models


class Flatten(nn.Module):
    """Flatten layer to convert 4D tensor to 2D tensor."""
    def forward(self, input):
        return input.view(input.size(0), -1)


class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network using ResNet18 as backbone.

    Args:
        contra_loss (bool): If True return raw embeddings (for contrastive/triplet loss).
                            If False apply FC head and return sigmoid similarity score.
    """
    def __init__(self, contra_loss=False):
        super(SiameseNetwork, self).__init__()
        self.contra_loss = contra_loss

        # ── Backbone ──────────────────────────────────────────────────
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv to accept RGB (3 channels) — already the default for
        # ResNet18, but we make it explicit and configurable.
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Store feature dimension (512 for ResNet18)
        self.feat_dim = backbone.fc.in_features  # 512

        # Remove final classification head: keep everything up to (and including)
        # the adaptive average pool, then flatten manually.
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1],   # drops the final Linear layer
            Flatten()
        )

        # ── Similarity head (BCE mode only) ───────────────────────────
        if not self.contra_loss:
            self.fc = nn.Sequential(
                nn.Linear(self.feat_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            self.fc.apply(self.init_weights)

    # ------------------------------------------------------------------
    def init_weights(self, m):
        """Xavier init for Linear layers; small positive bias."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.01)

    # ------------------------------------------------------------------
    def forward_once(self, x):
        """Extract feature vector for a single image."""
        return self.backbone(x)          # shape: (B, 512)

    # ------------------------------------------------------------------
    def forward(self, input1, input2):
        """
        Forward pass.

        BCE mode  → returns scalar similarity score per pair  (B, 1)
        Contrastive mode → returns (emb1, emb2) each of shape (B, 512)
        """
        emb1 = self.forward_once(input1)
        emb2 = self.forward_once(input2)

        if self.contra_loss:
            # Return raw embeddings so the contrastive / triplet loss can use distances
            return emb1, emb2
        else:
            # Concatenate and predict similarity
            combined = torch.cat([emb1, emb2], dim=1)   # (B, 1024)
            return self.fc(combined)                     # (B, 1)