"""把 train.py 生成的 metrics.json 转成 TensorBoard 事件文件，便于可视化历史训练。

train.py 加 --tensorboard 后，训练中会实时写入事件文件；本脚本用于给"没有开启
--tensorboard 的旧训练"补一份事件文件，无需重训即可在 TensorBoard 里查看曲线。

用法：
    py -3.11 export_tensorboard.py artifacts/run_all10
会在 <output_dir>/tensorboard 下生成事件文件。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="把 metrics.json 转为 TensorBoard 事件文件。")
    parser.add_argument("output_dir", type=Path, help="train.py 的 --output-dir，内含 metrics.json")
    args = parser.parse_args()

    metrics_path = args.output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"未找到 {metrics_path}")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    writer = SummaryWriter(args.output_dir / "tensorboard")

    # 每轮：训练损失 + 验证指标 + 学习率
    for row in data["history"]:
        epoch = row["epoch"]
        writer.add_scalar("train/loss_epoch", row["train_loss"], epoch)
        writer.add_scalar("val/loss", row["loss"], epoch)
        writer.add_scalar("val/accuracy", row["accuracy"], epoch)
        writer.add_scalar("val/precision", row["precision"], epoch)
        writer.add_scalar("val/recall", row["recall"], epoch)
        writer.add_scalar("lr", row["next_learning_rate"], epoch)

    # 最终测试集指标（放在最后一轮之后的位置）
    for key, value in data["test"].items():
        writer.add_scalar(f"test/{key}", value, len(data["history"]) + 1)

    writer.close()
    print(f"已写入 TensorBoard 事件到 {args.output_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
