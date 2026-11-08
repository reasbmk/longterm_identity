"""Quality-aware attention encoder and pairwise identity matcher."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class TrackletEncoder(nn.Module):
    def __init__(self, quality_dim: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.feature_dim = 512
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim + quality_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, images: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frames = images.shape[:2]
        features = self.backbone(images.flatten(0, 1)).view(batch_size, frames, self.feature_dim)
        features = F.normalize(features, dim=-1)
        logits = self.attention(torch.cat((features, quality), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=1)
        embedding = (features * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(embedding, dim=-1), weights


class PairMatcher(nn.Module):
    """Returns a logit that two same-class tracklets are one local identity."""

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        pair = torch.cat((first, second, torch.abs(first - second), first * second), dim=-1)
        return self.head(pair).squeeze(-1)


def contrastive_loss(first: torch.Tensor, second: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    distance = 1 - F.cosine_similarity(first, second)
    positive = labels * distance.square()
    negative = (1 - labels) * F.relu(margin - distance).square()
    return (positive + negative).mean()
