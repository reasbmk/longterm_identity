"""用训练好的模型在真实 crop 上做推理演示，生成 PPT 用的匹配效果图。

用法：
    py -3.11 demo_match.py

读取：
    artifacts/run_all10_e30/best.pt       训练好的模型（encoder + matcher）
    artifacts/tracklets_global_cached_224.jsonl   轨迹 manifest（含裁剪图路径）
输出到 artifacts/：
    fig_demo.png   若干对轨迹的裁剪图 + 模型预测（同一/不同 + 概率）

示例选取：
    正例 = 同一 global_inventory_id 出现在两条不同路线（跨路线识别同一物资）
    负例 = 同类别、不同物资编号（易混淆的相似物资）
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import PairMatcher, TrackletEncoder

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FRAMES = 8            # 与训练一致：每条轨迹取 8 帧
N_POS = 3             # 图里画几个正例 / 负例
N_NEG_CORRECT = 2     # 负例中"正确区分"的样例数
N_NEG_HARD = 0        # 负例中"困难样例（误判）"的样例数

CKPT = Path("artifacts/run_all10_e30/best.pt")
MANIFEST = Path("artifacts/tracklets_global_cached_224.jsonl")


def identity(record: dict) -> str:
    """轨迹的物资身份：优先全局编号，否则退回本地 identity_key。"""
    return record.get("global_inventory_id") or record["identity_key"]


def build_transform() -> transforms.Compose:
    # 与 dataset 评估态（training=False）完全一致：仅 resize + 归一化，无增强
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def encode(record: dict, encoder: TrackletEncoder, device: str, transform) -> torch.Tensor:
    """把一条轨迹编码成 512 维 embedding（取前 8 帧，与评估一致）。"""
    observations = record["observations"][:FRAMES]
    images, quality = [], []
    for obs in observations:
        with Image.open(obs["cached_crop_path"]) as src:
            images.append(transform(src.convert("RGB")))
        width, height = obs["bbox_xywh"][2:]
        sharpness = min(1.0, max(0.0, (width * height) / 40000.0))
        quality.append([obs["confidence"], obs["visibility"], sharpness])

    valid = torch.zeros(FRAMES, dtype=torch.bool)
    valid[: len(images)] = True
    while len(images) < FRAMES:          # 帧数不足时补全零图 + 零质量
        images.append(torch.zeros_like(images[0]))
        quality.append([0.0, 0.0, 0.0])

    images = torch.stack(images).unsqueeze(0).to(device)      # [1, 8, 3, 224, 224]
    quality = torch.tensor(quality, dtype=torch.float32).unsqueeze(0).to(device)
    valid = valid.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding, _ = encoder(images, quality, valid)
    return embedding                                            # [1, 512]


def predict(rec_a: dict, rec_b: dict, encoder, matcher, device: str, transform) -> float:
    """两轨迹 → 相似度概率（sigmoid 后的 logit，0~1，越接近 1 越像同一物资）。"""
    ea = encode(rec_a, encoder, device, transform)
    eb = encode(rec_b, encoder, device, transform)
    with torch.no_grad():
        logit = matcher(ea, eb)
    return float(torch.sigmoid(logit).item())


def display_crops(record: dict, n: int = 4) -> list[np.ndarray]:
    """均匀采样 n 帧 crop 用于展示，返回 RGB 数组列表（帧不足则补最后一帧）。"""
    obs = record["observations"]
    if len(obs) >= n:
        idx = [int(round(i * (len(obs) - 1) / (n - 1))) for i in range(n)]
    else:
        idx = list(range(len(obs)))
    crops = [np.array(Image.open(obs[i]["cached_crop_path"]).convert("RGB")) for i in idx]
    while len(crops) < n:
        crops.append(crops[-1])
    return crops


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)

    backbone = ck.get("args", {}).get("backbone", "resnet18")
    use_cfsa = ck.get("args", {}).get("cfsa", False)
    encoder = TrackletEncoder(pretrained=False, backbone=backbone, use_cfsa=use_cfsa)
    encoder.load_state_dict(ck["encoder"])
    encoder.to(device).eval()
    matcher = PairMatcher(embedding_dim=encoder.feature_dim)
    matcher.load_state_dict(ck["matcher"])
    matcher.to(device).eval()
    transform = build_transform()

    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_gid = defaultdict(set)        # gid -> {sequence_id...}
    gid_seqs = defaultdict(dict)     # gid -> {sequence_id: record}
    by_class = defaultdict(list)     # class_id -> [record...]
    for r in records:
        gid = r.get("global_inventory_id")
        if gid:
            by_gid[gid].add(r["sequence_id"])
            gid_seqs[gid][r["sequence_id"]] = r
        by_class[int(r["class_id"])].append(r)

    # 正例：同一物资、出现在 ≥2 条路线（跨路线识别同一物资）
    cross = sorted(
        (g for g, seqs in by_gid.items() if len(seqs) >= 2),
        key=lambda g: -max(len(gid_seqs[g][s]["observations"]) for s in by_gid[g]),
    )
    pos_pairs = []
    for g in cross:
        seqs = sorted(by_gid[g], key=lambda s: -len(gid_seqs[g][s]["observations"]))
        pos_pairs.append((gid_seqs[g][seqs[0]], gid_seqs[g][seqs[1]], 1, f"同一物资 {g} · 跨路线"))
        if len(pos_pairs) >= N_POS:
            break

    # 负例候选：同类别、不同物资编号（随机采样一批，再统一打分）
    random.seed(1)
    neg_candidates = []
    for cls, recs in by_class.items():
        if len(recs) < 2:
            continue
        for _ in range(25):
            a, b = random.sample(recs, 2)
            if identity(a) != identity(b):
                neg_candidates.append((a, b))
    # 按物资编号对去重
    seen_pairs, uniq = set(), []
    for a, b in neg_candidates:
        key = tuple(sorted((identity(a), identity(b))))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        uniq.append((a, b))
    neg_candidates = uniq

    # 对全部负例候选打分，统计真实负例准确率
    neg_scored = [(a, b, predict(a, b, encoder, matcher, device, transform))
                  for a, b in neg_candidates]
    neg_correct = [(a, b, p) for a, b, p in neg_scored if p < 0.5]
    neg_hard = [(a, b, p) for a, b, p in neg_scored if p >= 0.5]
    if neg_scored:
        print(f"负例（同类别·不同物资）共 {len(neg_scored)} 对："
              f"判对 {len(neg_correct)}，判错 {len(neg_hard)}"
              f"（负例准确率 {len(neg_correct) / len(neg_scored):.1%}）")

    # 负例展示：若干正确区分 + 若干困难样例（误判）
    neg_pairs = []
    for a, b, _ in neg_correct[:N_NEG_CORRECT]:
        neg_pairs.append((a, b, 0, f"同类别 · {identity(a)} vs {identity(b)}"))
    for a, b, _ in neg_hard[:N_NEG_HARD]:
        neg_pairs.append((a, b, 0, f"困难样例 · {identity(a)} vs {identity(b)}"))

    # 跑推理，组装 (rec_a, rec_b, label, note, prob)
    pairs = []
    for rec_a, rec_b, label, note in pos_pairs + neg_pairs:
        prob = predict(rec_a, rec_b, encoder, matcher, device, transform)
        pairs.append((rec_a, rec_b, label, note, prob))
        truth = "同一" if label == 1 else "不同"
        pred = "同一" if prob >= 0.5 else "不同"
        ok = "OK" if (prob >= 0.5) == (label == 1) else "XX"
        print(f"{ok} {note:32s}  真值={truth}  预测={pred} ({prob:.3f})")

    # ===== 画图：每行 = 轨迹A(4帧) | 中间标签 | 轨迹B(4帧) =====
    n = len(pairs)
    fig, axes = plt.subplots(
        n, 9, figsize=(20, 3.5 * n),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 2.1, 1, 1, 1, 1]},
    )
    if n == 1:
        axes = axes[None, :]

    for i, (rec_a, rec_b, label, note, prob) in enumerate(pairs):
        ca = display_crops(rec_a, 4)
        cb = display_crops(rec_b, 4)
        for j in range(4):
            axes[i, j].imshow(ca[j]); axes[i, j].axis("off")
            axes[i, 5 + j].imshow(cb[j]); axes[i, 5 + j].axis("off")
        axes[i, 0].set_title(f"轨迹A\nseq {rec_a['sequence_id']}", loc="left",
                             fontsize=12, color="#2E86C1")
        axes[i, 8].set_title(f"轨迹B\nseq {rec_b['sequence_id']}", loc="right",
                             fontsize=12, color="#2E86C1")

        axl = axes[i, 4]; axl.axis("off")
        truth = "同一物资" if label == 1 else "不同物资"
        pred = "同一" if prob >= 0.5 else "不同"
        color = "#1E8449" if (prob >= 0.5) == (label == 1) else "#C0392B"
        axl.text(0.5, 0.84, note, ha="center", va="center", fontsize=13, color="#7F8C8D")
        axl.text(0.5, 0.64, "真值", ha="center", va="center", fontsize=13, color="#7F8C8D")
        axl.text(0.5, 0.47, truth, ha="center", va="center", fontsize=19,
                 fontweight="bold", color="#34495E")
        axl.text(0.5, 0.27, "预测", ha="center", va="center", fontsize=13, color="#7F8C8D")
        axl.text(0.5, 0.10, f"{pred}  {prob:.2f}", ha="center", va="center", fontsize=19,
                 fontweight="bold", color=color)

    fig.suptitle("模型演示：TrackletEncoder + PairMatcher 对真实轨迹的匹配预测"
                 "（绿色=预测正确，红色=预测错误）", y=0.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = Path("artifacts/fig_demo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n已生成：{out}")


if __name__ == "__main__":
    main()
