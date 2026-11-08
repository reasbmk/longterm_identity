"""根据 train.py 生成的 metrics.json 画训练曲线，用于阶段汇报 PPT。

用法：
    py -3.11 plot_training.py
会读取 artifacts/run_all10 与 artifacts/run_all10_e30 的 metrics.json，
在 artifacts/ 下输出：
    fig_compare.png   20 轮 vs 30 轮的验证指标对比
    fig_detail.png    30 轮训练的完整曲线（loss/acc/prec/recall/lr）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows 下用 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    artifacts = Path("artifacts")
    runs = {
        "20 轮 (batch16)": artifacts / "run_all10" / "metrics.json",
        "30 轮 (batch16)": artifacts / "run_all10_e30" / "metrics.json",
    }
    data = {name: load_metrics(p) for name, p in runs.items() if p.is_file()}

    # 图 1：20 vs 30 轮 验证指标对比（同一 epoch 轴上叠加）
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, m in data.items():
        epochs = [r["epoch"] for r in m["history"]]
        for ax, key, title in (
            (axes[0], "accuracy", "验证集 accuracy"),
            (axes[1], "precision", "验证集 precision"),
            (axes[2], "recall", "验证集 recall"),
        ):
            ax.plot(epochs, [r[key] for r in m["history"]], marker="o", ms=4, label=name)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("不同训练轮数对比（验证集，按物资隔离划分）", y=1.02)
    fig.tight_layout()
    fig.savefig(artifacts / "fig_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 2：30 轮训练的完整曲线
    m30 = data.get("30 轮 (batch16)")
    if m30:
        hist = m30["history"]
        epochs = [r["epoch"] for r in hist]
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        panels = [
            ("train_loss", "train loss", axes[0, 0]),
            ("loss", "val loss", axes[0, 1]),
            ("accuracy", "val accuracy", axes[0, 2]),
            ("precision", "val precision", axes[1, 0]),
            ("recall", "val recall", axes[1, 1]),
        ]
        for key, title, ax in panels:
            ax.plot(epochs, [r[key] for r in hist], marker="o", ms=4)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        # 学习率单独画（对数坐标，展示余弦退火）
        ax = axes[1, 2]
        ax.plot(epochs, [r["next_learning_rate"] for r in hist], marker="o", ms=4, color="purple")
        ax.set_title("学习率（余弦退火）")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        fig.suptitle("30 轮训练完整曲线", y=1.0)
        fig.tight_layout()
        fig.savefig(artifacts / "fig_detail.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("已生成：")
    for f in ("fig_compare.png", "fig_detail.png"):
        if (artifacts / f).is_file():
            print("  artifacts/" + f)


if __name__ == "__main__":
    main()
