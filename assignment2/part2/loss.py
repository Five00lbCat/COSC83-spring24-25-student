"""
Loss functions for Siamese Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss.

    L = (1 - Y) * 0.5 * D^2  +  Y * 0.5 * max(0, margin - D)^2

    Convention used here (matching the dataset labels):
        label = 1  →  SIMILAR   pair  (pull together)
        label = 0  →  DISSIMILAR pair (push apart)

    Args:
        margin (float): Minimum desired distance for dissimilar pairs.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)

        # Similar pairs: minimise distance
        loss_sim   = label * 0.5 * torch.pow(dist, 2)
        # Dissimilar pairs: push distance beyond margin
        loss_disim = (1 - label) * 0.5 * torch.pow(
            torch.clamp(self.margin - dist, min=0.0), 2
        )
        return torch.mean(loss_sim + loss_disim)


class TripletLoss(nn.Module):
    """
    Triplet loss.

    L = max(0, D(anchor, positive) - D(anchor, negative) + margin)

    Args:
        margin (float): Separation margin between positive and negative distances.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self._loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self._loss_fn(anchor, positive, negative)