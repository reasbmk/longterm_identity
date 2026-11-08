"""Datasets for same-class local-identity tracklet matching."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identity_of(record: dict) -> str:
    """返回跨路线全局编号；未标注时回退到序列内 identity_key。"""
    return record.get("global_inventory_id") or record["identity_key"]


class TrackletPairDataset(Dataset):
    def __init__(self, records: list[dict], frames_per_tracklet: int, training: bool, image_size: int = 224,
                 gid_to_class: dict[str, int] | None = None) -> None:
        self.records = records
        self.frames_per_tracklet = frames_per_tracklet
        self.training = training
        self.image_size = image_size
        # 全局编号 -> 整数类别索引，供 InfoNCE / ArcFace / CenterLoss 等辅助损失使用
        self.gid_to_class = gid_to_class or {}
        self.by_identity: dict[str, list[int]] = defaultdict(list)
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.by_identity[identity_of(record)].append(index)
            self.by_class[int(record["class_id"])].append(index)
        self.positive_anchors = [index for index, record in enumerate(records) if len(self.by_identity[identity_of(record)]) > 1]
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        augmentations = [transforms.Resize((image_size, image_size))]
        if training:
            augmentations.extend([transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.08, 0.03)])
        augmentations.extend([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        self.transform = transforms.Compose(augmentations)

    def __len__(self) -> int:
        return max(len(self.positive_anchors), len(self.records))

    def _sample_pair(self, index: int) -> tuple[dict, dict, float]:
        anchor = self.records[self.positive_anchors[index % len(self.positive_anchors)]]
        anchor_identity = identity_of(anchor)
        is_positive = bool(random.getrandbits(1)) if self.training else index % 2 == 0
        if is_positive:
            # 正样本：同全局编号（可跨序列）、不同 tracklet
            choices = [item for item in self.by_identity[anchor_identity] if self.records[item]["tracklet_id"] != anchor["tracklet_id"]]
            if choices:
                return anchor, self.records[random.choice(choices)], 1.0
        # 负样本：同类别、不同全局编号；优先同序列（场景相同，更难）
        choices = [
            item for item in self.by_class[int(anchor["class_id"])]
            if identity_of(self.records[item]) != anchor_identity
        ]
        if not choices:
            # 少数类别按 global_id 划分后可能只留下一个物资编号，无法构造同类别负样本；
            # 退化为跨类别负样本（仍保证 label=0 可用），避免评估阶段直接崩溃。
            choices = [
                item for item in range(len(self.records))
                if identity_of(self.records[item]) != anchor_identity
            ]
        if not choices:
            raise RuntimeError("Need at least two identities to create negative pairs")
        same_sequence = [item for item in choices if self.records[item]["sequence_id"] == anchor["sequence_id"]]
        return anchor, self.records[random.choice(same_sequence or choices)], 0.0

    def _read_tracklet(self, record: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = record["observations"]
        if self.training:
            selected = random.sample(observations, k=min(len(observations), self.frames_per_tracklet))
        else:
            selected = observations[: self.frames_per_tracklet]
        images, quality = [], []
        for observation in selected:
            cached_path = observation.get("cached_crop_path")
            if cached_path:
                with Image.open(cached_path) as source:
                    images.append(self.transform(source.convert("RGB")))
                width, height = observation["bbox_xywh"][2:]
            else:
                with Image.open(observation["image_path"]) as source:
                    image = source.convert("RGB")
                    x, y, width, height = observation["bbox_xywh"]
                    left, top = max(0, int(x)), max(0, int(y))
                    right, bottom = min(image.width, int(x + width)), min(image.height, int(y + height))
                    crop = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
                    images.append(self.transform(crop))
            sharpness = min(1.0, max(0.0, (width * height) / 40000.0))
            quality.append([observation["confidence"], observation["visibility"], sharpness])

        valid = torch.zeros(self.frames_per_tracklet, dtype=torch.bool)
        valid[: len(images)] = True
        while len(images) < self.frames_per_tracklet:
            images.append(torch.zeros_like(images[0]))
            quality.append([0.0, 0.0, 0.0])
        return torch.stack(images), torch.tensor(quality, dtype=torch.float32), valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        first, second, label = self._sample_pair(index)
        first_data = self._read_tracklet(first)
        second_data = self._read_tracklet(second)
        return {
            "first_images": first_data[0], "first_quality": first_data[1], "first_valid": first_data[2],
            "second_images": second_data[0], "second_quality": second_data[1], "second_valid": second_data[2],
            "label": torch.tensor(label, dtype=torch.float32),
            # 两段轨迹各自的类别索引（用于辅助对比/分类损失）；缺失时为 -1
            "first_class": torch.tensor(self.gid_to_class.get(identity_of(first), -1), dtype=torch.long),
            "second_class": torch.tensor(self.gid_to_class.get(identity_of(second), -1), dtype=torch.long),
        }
