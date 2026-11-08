"""画两张汇报用示意图：整体流程图 + 模型结构图。

用法：
    py -3.11 draw_diagrams.py
输出到 artifacts/：
    fig_pipeline.png   检测→跟踪→Re-ID 全流程
    fig_model.png      TrackletEncoder + PairMatcher 结构
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色：输入/已有=蓝，本次核心=橙，输出=绿
C_INPUT = ("#EAF2F8", "#2E86C1")
C_DONE = ("#D6EAF8", "#2E86C1")
C_NEW = ("#FDEBD0", "#E67E22")
C_OUT = ("#D5F5E3", "#1E8449")
C_GRAY = ("#F2F3F4", "#7F8C8D")


def box(ax, x, y, w, h, text, fc, ec, fs=11, bold_first=True):
    """画一个圆角矩形，text 首行加粗作为标题。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=1.6, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2, y + h / 2 + (len(lines) - 1) * 0.08 * fs,
                "\n".join(lines), ha="center", va="center", fontsize=fs,
                color="#1B2631", linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#1B2631", linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, color="#34495E"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=16, linewidth=1.8, color=color))


def draw_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4)
    ax.axis("off")

    stages = [
        ("图像序列 / 视频\n(无人机拍摄)", C_INPUT, "输入"),
        ("YOLO 检测\n框 + 类别 + 置信度", C_DONE, "已有"),
        ("BoT-SORT 跟踪\n跨帧 ID 绑定", C_DONE, "已有"),
        ("轨迹切分\ntracklet", C_NEW, "本次"),
        ("Re-ID 身份匹配\n跨路线识别同一物资", C_NEW, "本次核心"),
        ("全局库存 ID\n(global_inventory_id)", C_OUT, "输出"),
    ]
    x, y, w, h = 0.4, 1.6, 2.3, 1.6
    gap = 0.35
    centers = []
    for i, (txt, (fc, ec), tag) in enumerate(stages):
        xi = x + i * (w + gap)
        box(ax, xi, y, w, h, txt, fc, ec, fs=12)
        ax.text(xi + w / 2, y - 0.32, tag, ha="center", va="top", fontsize=11,
                color=ec, fontweight="bold")
        centers.append((xi + w, y + h / 2))
    for i in range(len(stages) - 1):
        arrow(ax, centers[i][0], centers[i][1], centers[i][0] + gap, centers[i][1])

    fig.suptitle("长期库存身份匹配 · 全流程（检测 → 跟踪 → 重识别）", y=0.97, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_model(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(0, 12); ax.set_ylim(0, 13)
    ax.axis("off")

    # ===== 两条轨迹输入（并排）=====
    box(ax, 0.8, 10.8, 4.2, 1.6, "轨迹1 (tracklet1)\n8 帧 crop + 质量", *C_INPUT, fs=11)
    box(ax, 7.0, 10.8, 4.2, 1.6, "轨迹2 (tracklet2)\n8 帧 crop + 质量", *C_INPUT, fs=11)

    # ===== TrackletEncoder（共享权重，展开内部）=====
    box(ax, 0.8, 7.4, 10.4, 2.2,
        "TrackletEncoder（共享权重 · 两条轨迹各编码一次）\n"
        "ResNet-18 逐帧提特征 → 质量注意力池化 → 512 维 embedding（L2 归一化）",
        *C_DONE, fs=11)

    # ===== 两个 embedding =====
    box(ax, 2.0, 5.0, 3.0, 1.3, "f1（512 维）", *C_NEW, fs=12)
    box(ax, 7.0, 5.0, 3.0, 1.3, "f2（512 维）", *C_NEW, fs=12)

    # ===== PairMatcher =====
    box(ax, 0.8, 2.6, 10.4, 1.6,
        "PairMatcher\nconcat[f1, f2, |f1-f2|, f1*f2]（2048 维）→ MLP（512 → 128 → 1）",
        *C_NEW, fs=11)

    # ===== 最终输出（绿色高亮，位于最底部）=====
    box(ax, 4.0, 0.6, 4.0, 1.3, "最终输出\n同一 / 不同", *C_OUT, fs=13)

    # 箭头：两条轨迹 → encoder
    arrow(ax, 2.9, 10.8, 3.0, 9.6)
    arrow(ax, 9.1, 10.8, 9.0, 9.6)
    # encoder → 两个 embedding
    arrow(ax, 3.0, 7.4, 3.5, 6.3)
    arrow(ax, 9.0, 7.4, 8.5, 6.3)
    # embedding → matcher
    arrow(ax, 3.5, 5.0, 3.0, 4.2)
    arrow(ax, 8.5, 5.0, 9.0, 4.2)
    # matcher → 最终输出
    arrow(ax, 6.0, 2.6, 6.0, 1.9)

    fig.suptitle("模型结构：TrackletEncoder + PairMatcher（最终输出在底部）", y=0.98,
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    artifacts = Path("artifacts")
    draw_pipeline(artifacts / "fig_pipeline.png")
    draw_model(artifacts / "fig_model.png")
    print("已生成：")
    for f in ("fig_pipeline.png", "fig_model.png"):
        if (artifacts / f).is_file():
            print("  artifacts/" + f)


if __name__ == "__main__":
    main()
