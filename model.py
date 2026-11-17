"""Quality-aware attention encoder and pairwise identity matcher."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
)

# 可选骨干网络：(构建函数, 预训练权重枚举, 输出特征维数)
BACKBONES: dict[str, tuple] = {
    "resnet18": (resnet18, ResNet18_Weights, 512),
    "resnet34": (resnet34, ResNet34_Weights, 512),
    "resnet50": (resnet50, ResNet50_Weights, 2048),
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights, 576),
}


class CrossFrameStabilityAttention(nn.Module):
    """跨帧稳定性注意力（CFSA，自创模块）。

    针对"相似物资误识别"：同一件物资在轨迹里会反复出现，真正属于物体的区域是
    **跨帧稳定**的（真实部件），而背景/反光/遮挡只出现在个别帧、是**不稳定**的。
    本模块对 F 帧的空间特征图做两路加权：
      (1) 通道判别性 —— SE 式 MLP 学习哪些通道对"区分不同物资"更重要；
      (2) 空间稳定性 —— 每个空间位置在相邻帧间的余弦一致性越高，权重越大。
    两者相乘得到时空显著性，把空间特征聚合成每帧的判别性向量。

    输入 x:     [B, F, C, H, W]（F 帧、C 通道、H×W 空间）
    输入 valid: [B, F]（哪些帧有效，无效帧为补齐的零图）
    输出:       [B, F, C]（每帧聚合后的特征，后续仍走质量注意力做帧间加权）
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # 通道判别性权重（SE 风格）：跨帧+空间平均后经两层 MLP 得到 [0,1] 通道门控
        self.channel_se = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        # 空间稳定性的可学习温度：softmax 前缩放，控制注意力锐度
        self.stability_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        valid_f = valid.float().view(batch, frames, 1, 1, 1)  # [B, F, 1, 1, 1]

        # ---- 通道判别性：只对有效帧做跨帧+空间平均，避免补齐零图污染统计 ----
        channel_stat = (x * valid_f).sum(dim=(1, 3, 4)) / valid_f.sum(dim=(1, 3, 4)).clamp(min=1.0)
        channel_weight = self.channel_se(channel_stat)                    # [B, C]
        x_c = x * channel_weight[:, None, :, None, None]                  # 通道重加权

        # ---- 空间稳定性：相邻帧在同一空间位置的余弦一致性 ----
        x_n = F.normalize(x_c, dim=2)                                     # 通道方向归一化 → 余弦
        similarity = (x_n[:, :-1] * x_n[:, 1:]).sum(dim=2)                # [B, F-1, H, W]
        pair_valid = (valid[:, :-1] & valid[:, 1:]).float()               # [B, F-1] 相邻帧都有效
        similarity = similarity * pair_valid[:, :, None, None]
        stability = similarity.sum(dim=1) / pair_valid.sum(dim=1).clamp(min=1.0)[:, None, None]  # [B, H, W]
        spatial_weight = torch.softmax(
            (stability * self.stability_temp).flatten(1), dim=1
        ).view(batch, height, width)                                      # [B, H, W]

        # ---- 用空间稳定性权重聚合，得到每帧特征 [B, F, C] ----
        return (x_c * spatial_weight[:, None, None]).sum(dim=(3, 4))


class TrackletEncoder(nn.Module):
    def __init__(self, quality_dim: int = 3, pretrained: bool = False, backbone: str = "resnet18", use_cfsa: bool = False) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"未知 backbone：{backbone}，可选 {sorted(BACKBONES)}")
        builder, weights_enum, feature_dim = BACKBONES[backbone]
        weights = weights_enum.DEFAULT if pretrained else None
        self.backbone = builder(weights=weights)
        # 去掉分类头：ResNet 系列用 fc，MobileNetV3 用 classifier
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        self.feature_dim = feature_dim
        self.use_cfsa = use_cfsa
        if use_cfsa:
            # CFSA 需要空间特征图：取 backbone 卷积部分（去掉最后的池化+分类头）。
            # children() 返回的是引用，参数与 self.backbone 共享，不会重复存储。
            self.spatial_backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.cfsa = CrossFrameStabilityAttention(feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim + quality_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, images: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frames = images.shape[:2]
        if self.use_cfsa:
            # 先取空间特征图，再用 CFSA 聚合出每帧判别性向量 [B, F, C]
            spatial = self.spatial_backbone(images.flatten(0, 1))  # [B*F, C, H, W]
            _, channels, height, width = spatial.shape
            spatial = spatial.view(batch_size, frames, channels, height, width)
            features = self.cfsa(spatial, valid)                  # [B, F, C]
        else:
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
