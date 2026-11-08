"""Loss functions adapted from the palm-vein ReID baseline (pv_baseline/loss.py).

原项目是闭集 ReID（人脸/掌纹），loss.py 提供两类损失：
  - 闭集分类头：ArcFace / CosFace / CenterLoss（带可学习类别权重 / 类中心）
  - 批内对比：InfoNCE / focal NCE（对归一化嵌入做温度缩放 + 难样本加权）

本项目是「轨迹对匹配器」：主损失是配对二分类 BCE（PairMatcher 输出 logit，
判断两段轨迹是否同一件物资），嵌入由 TrackletEncoder 输出（L2 归一化 512 维）。
本模块把上述损失适配为**辅助损失**，作用在嵌入上做正则：

  - `focal_bce_loss`    : 对主 BCE 做 focal 加权（只改主损失，无需类别 id）
  - `infonce_loss`      : 监督对比（SupCon 式，多正样本、log-sum-exp 稳定）
  - `focal_infonce_loss`: focal InfoNCE（对应参考实现的 Focal_Nce_v3）
  - `ArcFace` / `CosFace`: 加性/角度 margin 闭集分类头（需要 learnable 权重）
  - `CenterLoss`        : 拉向类中心

所有辅助损失统一接口：接收 [N, D] 归一化嵌入 + [N] 整数类别 id
（global_inventory_id 的映射索引），与批内配对采样方式解耦。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def focal_bce_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal 二分类交叉熵：对易分对的损失按 (1 - p_t)^gamma 降权，聚焦难分对。

    logits: [N] 配对 logit；labels: [N] 0/1。
    """
    probabilities = torch.sigmoid(logits)
    # p_t = p if y==1 else 1-p
    p_t = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    focal_weight = (1.0 - p_t).pow(gamma)
    elementwise = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (focal_weight * elementwise).mean()


def infonce_loss(embeddings: torch.Tensor, class_ids: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """监督对比损失（SupCon）：同 id 的嵌入拉近，不同 id 的嵌入推远。

    embeddings: [N, D] 已 L2 归一化；class_ids: [N] 整数类别 id。
    每个 query 对同 id 的所有其它样本取平均（支持同 id 出现多次）。
    """
    device = embeddings.device
    num = embeddings.size(0)
    # 余弦相似度矩阵，除以温度
    similarity = (embeddings @ embeddings.t()) / temperature
    # 正样本掩码：同 id 且非自身
    ids_equal = class_ids.unsqueeze(0) == class_ids.unsqueeze(1)
    pos_mask = ids_equal & (~torch.eye(num, dtype=torch.bool, device=device))
    # 数值稳定：每行减去最大值后再 exp（log-sum-exp）
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - logits.exp().sum(dim=1, keepdim=True).log()
    # 每个 query 对所有正样本的 log 概率取平均
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()


def focal_infonce_loss(
    embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    temperature: float = 0.07,
    gamma: float = 2.0,
    keep_weight: float = 1.0,
) -> torch.Tensor:
    """focal InfoNCE：在 InfoNCE 上对难样本（低正样本概率）额外加权。

    对应参考实现 Focal_Nce_v3：probability = sum_pos / (sum_pos + sum_neg)，
    focal_weight = (1 - probability)^gamma + keep_weight。此处用逐行减最大值
    避免参考实现里的 exp 溢出（减常数不改变比值）。
    """
    device = embeddings.device
    num = embeddings.size(0)
    similarity = (embeddings @ embeddings.t()).clamp(-1.0, 1.0)
    similarity = (similarity - similarity.max(dim=1, keepdim=True).values.detach()) / temperature
    exp = similarity.exp()
    ids_equal = class_ids.unsqueeze(0) == class_ids.unsqueeze(1)
    eye = torch.eye(num, dtype=torch.bool, device=device)
    pos = ids_equal & ~eye
    neg = ~ids_equal & ~eye
    logit_pos = (exp * pos).sum(dim=1)
    logit_neg = (exp * neg).sum(dim=1)
    probability = logit_pos / (logit_pos + logit_neg + 1e-8)
    nce = -torch.log(probability + 1e-8)
    focal_weight = (1.0 - probability).pow(gamma) + keep_weight
    return (focal_weight * nce).mean()


class ArcFace(nn.Module):
    """加性角度 margin 分类头（参考 ArcFace）。"""

    def __init__(self, num_classes: int, feat_dim: int, s: float = 64.0, m: float = 0.35) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """embeddings: [N, D] L2 归一化；class_ids: [N] 整数类别 id。"""
        # 类别权重也归一化（NormFace 做法），余弦即 mm(emb, W^T)
        weight = F.normalize(self.weight, dim=1)
        cosine = (embeddings @ weight.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = F.one_hot(class_ids, num_classes=self.weight.size(0)).float()
        theta = cosine.acos()
        # 只对真值类别加角度 margin，再夹到 [0, pi]
        theta_m = (theta + self.m * one_hot).clamp(max=math.pi)
        logits = theta_m.cos() * self.s
        return F.cross_entropy(logits, class_ids)


class CosFace(nn.Module):
    """加性余弦 margin 分类头（参考 CosFace）。"""

    def __init__(self, num_classes: int, feat_dim: int, s: float = 64.0, m: float = 0.4) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        cosine = (embeddings @ weight.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = F.one_hot(class_ids, num_classes=self.weight.size(0)).float()
        # 真值类别余弦减 margin，其余不变
        phi = cosine - self.m
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(output * self.s, class_ids)


class CenterLoss(nn.Module):
    """中心损失：把同类嵌入拉向对应类中心（参考 CenterLoss）。"""

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_normal_(self.centers)

    def forward(self, embeddings: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        centers = self.centers[class_ids]
        return ((embeddings - centers).pow(2).sum(dim=1)).mean()
