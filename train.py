"""Train the quality-aware attention tracklet matcher."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TrackletPairDataset, identity_of, load_manifest
from losses import (ArcFace, CenterLoss, CosFace, focal_bce_loss,
                    focal_infonce_loss, infonce_loss)
from model import PairMatcher, TrackletEncoder, contrastive_loss


def split_records_by_sequence(records: list[dict], val_sequences: set[str], test_sequences: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """按序列划分 train/val/test（无全局编号时的回退方案）。"""
    train, validation, test = [], [], []
    for record in records:
        sequence = record["sequence_id"]
        if sequence in test_sequences:
            test.append(record)
        elif sequence in val_sequences:
            validation.append(record)
        else:
            train.append(record)
    return train, validation, test


def split_records_by_global_id(records: list[dict], val_fraction: float, test_fraction: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """按 global_inventory_id 划分，同一件物资的所有 tracklet 只进一个集合。

    跨路线的同一物资会出现在多条序列里，若仍按序列划分，train/val/test 会
    共享同一件物资，验证指标会虚高。这里按物资编号随机划入 train/val/test。
    """
    rng = random.Random(seed)
    ids = sorted({record["global_inventory_id"] for record in records if record.get("global_inventory_id")})
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    n_val = max(1, int(round(len(ids) * val_fraction)))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train, validation, test = [], [], []
    for record in records:
        global_id = record.get("global_inventory_id")
        if global_id in test_ids:
            test.append(record)
        elif global_id in val_ids:
            validation.append(record)
        else:
            train.append(record)
    return train, validation, test


def evaluate(encoder: TrackletEncoder, matcher: PairMatcher, loader: DataLoader, device: torch.device) -> dict[str, float]:
    encoder.eval()
    matcher.eval()
    labels, predictions = [], []
    total_loss, total_samples = 0.0, 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch in loader:
            first = encoder(batch["first_images"].to(device), batch["first_quality"].to(device), batch["first_valid"].to(device))[0]
            second = encoder(batch["second_images"].to(device), batch["second_quality"].to(device), batch["second_valid"].to(device))[0]
            targets = batch["label"].to(device)
            logits = matcher(first, second)
            loss = criterion(logits, targets) + 0.2 * contrastive_loss(first, second, targets)
            total_loss += loss.item() * targets.numel()
            total_samples += targets.numel()
            probabilities = torch.sigmoid(logits)
            labels.extend(batch["label"].tolist())
            predictions.extend(probabilities.cpu().tolist())
    if not labels:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    binary = [value >= 0.5 for value in predictions]
    correct = sum(int(prediction == bool(label)) for prediction, label in zip(binary, labels))
    tp = sum(int(prediction and label > 0.5) for prediction, label in zip(binary, labels))
    fp = sum(int(prediction and label <= 0.5) for prediction, label in zip(binary, labels))
    fn = sum(int(not prediction and label > 0.5) for prediction, label in zip(binary, labels))
    return {
        "loss": total_loss / total_samples,
        "accuracy": correct / len(labels),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def compute_auxiliary_loss(
    args,
    aux_loss: nn.Module | None,
    first: torch.Tensor,
    second: torch.Tensor,
    first_class: torch.Tensor,
    second_class: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """按 --loss 计算辅助损失，返回 device 上的标量张量。

    first/second: 两段轨迹的 L2 归一化嵌入 [B, D]；labels: [B] 0/1 配对标签；
    first_class/second_class: 各自轨迹的全局编号类别索引 [B]。
    """
    zero = torch.zeros((), device=device)
    if args.loss == "none":
        return zero
    if args.loss == "contrastive":
        return contrastive_loss(first, second, labels)
    # 其余辅助损失都需要批内嵌入 + 类别索引（InfoNCE 用整批做正负样本）
    embeddings = torch.cat((first, second), dim=0)
    class_ids = torch.cat((first_class, second_class)).to(device)
    if args.loss == "infonce":
        return infonce_loss(embeddings, class_ids, args.temperature)
    if args.loss == "focal_nce":
        return focal_infonce_loss(embeddings, class_ids, args.temperature, args.focal_gamma)
    if args.loss in ("arcface", "cosface", "center"):
        valid = class_ids >= 0
        if not valid.any():
            return zero
        return aux_loss(embeddings[valid], class_ids[valid])
    return zero


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cross-route inventory tracklet matcher.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frames-per-tracklet", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224, help="Square crop size fed into ResNet-18.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6, help="Final learning rate for cosine decay.")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--resume", type=Path, default=None, help="Load model weights from a previous best.pt or last.pt.")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="只训练这些类别（1..10）；省略则全部。")
    parser.add_argument("--split-by", choices=["sequence", "global_id"], default="global_id",
                        help="按序列还是按 global_inventory_id 划分 train/val/test。")
    parser.add_argument("--val-sequences", nargs="*", default=["008"])
    parser.add_argument("--test-sequences", nargs="*", default=["012"])
    parser.add_argument("--val-fraction", type=float, default=0.1, help="global_id 划分时验证集物资占比。")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="global_id 划分时测试集物资占比。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20, help="Print training progress every N batches.")
    parser.add_argument("--loss", choices=["contrastive", "infonce", "focal_nce", "arcface", "cosface", "center", "none"],
                        default="contrastive", help="辅助损失类型（主配对 BCE 始终保留）。")
    parser.add_argument("--loss-weight", type=float, default=0.2, help="辅助损失权重。")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE / focal NCE 的温度。")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="focal BCE / focal NCE 的 gamma。")
    parser.add_argument("--margin", type=float, default=0.35, help="ArcFace / CosFace 的 margin。")
    parser.add_argument("--scale", type=float, default=64.0, help="ArcFace / CosFace 的 scale。")
    parser.add_argument("--focal-bce", action="store_true", help="主配对 BCE 改用 focal BCE。")
    parser.add_argument("--tensorboard", action="store_true", help="在 output_dir/tensorboard 下写入 TensorBoard 事件文件。")
    args = parser.parse_args()
    if set(args.val_sequences) & set(args.test_sequences):
        raise ValueError("Validation and test sequences must not overlap")
    if args.image_size < 64:
        raise ValueError("image-size must be at least 64 pixels")
    if args.learning_rate <= 0 or args.min_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_manifest(args.manifest)
    if args.classes:
        records = [r for r in records if int(r["class_id"]) in args.classes]
    # 全局编号 -> 整数类别索引：辅助损失（InfoNCE/ArcFace/CenterLoss）需要整数 id。
    # 在划分前对全部 records 建立映射，保证 train/val/test 的索引一致。
    gid_to_class = {gid: index for index, gid in enumerate(sorted({identity_of(r) for r in records}))}
    num_classes = len(gid_to_class)
    if args.split_by == "global_id":
        if not any(r.get("global_inventory_id") for r in records):
            raise ValueError("--split-by global_id 需要 manifest 里有 global_inventory_id；请用 build_tracklets.py --global-id-csv 重新构建")
        train_records, validation_records, test_records = split_records_by_global_id(records, args.val_fraction, args.test_fraction, args.seed)
    else:
        train_records, validation_records, test_records = split_records_by_sequence(records, set(args.val_sequences), set(args.test_sequences))
    if not train_records or not validation_records or not test_records:
        raise ValueError("Each train/validation/test split must contain tracklets")
    train_data = TrackletPairDataset(train_records, args.frames_per_tracklet, training=True, image_size=args.image_size, gid_to_class=gid_to_class)
    validation_data = TrackletPairDataset(validation_records, args.frames_per_tracklet, training=False, image_size=args.image_size, gid_to_class=gid_to_class)
    test_data = TrackletPairDataset(test_records, args.frames_per_tracklet, training=False, image_size=args.image_size, gid_to_class=gid_to_class)
    loaders = {
        "train": DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available()),
        "validation": DataLoader(validation_data, args.batch_size, shuffle=False, num_workers=args.workers),
        "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=args.workers),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # A resume checkpoint already contains the image encoder weights, so it does
    # not need to create/download a fresh ImageNet-initialized backbone first.
    encoder = TrackletEncoder(pretrained=args.pretrained and args.resume is None).to(device)
    matcher = PairMatcher().to(device)
    # 闭集分类辅助头（ArcFace/CosFace/CenterLoss）需要可学习参数，随主模型一起优化
    aux_loss = None
    if args.loss == "arcface":
        aux_loss = ArcFace(num_classes, encoder.feature_dim, s=args.scale, m=args.margin).to(device)
    elif args.loss == "cosface":
        aux_loss = CosFace(num_classes, encoder.feature_dim, s=args.scale, m=args.margin).to(device)
    elif args.loss == "center":
        aux_loss = CenterLoss(num_classes, encoder.feature_dim).to(device)
    resume_checkpoint = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(resume_checkpoint["encoder"])
        matcher.load_state_dict(resume_checkpoint["matcher"])
        if aux_loss is not None and resume_checkpoint.get("aux_loss") is not None:
            aux_loss.load_state_dict(resume_checkpoint["aux_loss"])
    parameters = list(encoder.parameters()) + list(matcher.parameters())
    if aux_loss is not None:
        parameters += list(aux_loss.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_learning_rate,
        )
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        # Only restore optimizer/scheduler when resuming an interrupted run with
        # compatible hyperparameters. A best.pt fine-tune starts a fresh schedule.
        if resume_checkpoint.get("args", {}).get("learning_rate") == args.learning_rate:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            if scheduler is not None and resume_checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(resume_checkpoint["scheduler"])
    criterion = nn.BCEWithLogitsLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # TensorBoard 日志写入器：仅在 --tensorboard 开启时创建；延迟导入避免未安装时影响训练
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise SystemExit("--tensorboard 需要先安装 tensorboard：pip install tensorboard") from exc
        writer = SummaryWriter(args.output_dir / "tensorboard")
    best_score, history = -1.0, []
    if args.resume is not None:
        prior_metrics = args.resume.parent / "metrics.json"
        if prior_metrics.is_file():
            prior_history = json.loads(prior_metrics.read_text(encoding="utf-8")).get("history", [])
            if prior_history:
                best_score = max(float(row["precision"]) for row in prior_history)
        # Carry the prior best forward. If no fine-tuning epoch improves it, the
        # final test remains tied to the known best checkpoint rather than a
        # weaker final epoch.
        if not (args.output_dir / "best.pt").exists():
            shutil.copy2(args.resume, args.output_dir / "best.pt")

    print("=" * 72, flush=True)
    print("Long-term identity matcher training", flush=True)
    print(f"device={device} | cuda={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'disabled'}", flush=True)
    print(
        f"tracklets: train={len(train_records)}, val={len(validation_records)}, test={len(test_records)} | "
        f"positive anchors: train={len(train_data.positive_anchors)}, val={len(validation_data.positive_anchors)}, "
        f"test={len(test_data.positive_anchors)}",
        flush=True,
    )
    print(
        f"model: ResNet18(pretrained={args.pretrained}) + quality attention + pair MLP | "
        f"frames/tracklet={args.frames_per_tracklet}, crop={args.image_size}x{args.image_size}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, optimizer=AdamW, scheduler={args.scheduler}",
        flush=True,
    )
    print(
        f"loss: BCE{' (focal γ=' + str(args.focal_gamma) + ')' if args.focal_bce else ''} "
        f"+ {args.loss_weight} * {args.loss}"
        f"{' (T=' + str(args.temperature) + ')' if args.loss in ('infonce', 'focal_nce') else ''}"
        f"{' (s=' + str(args.scale) + ', m=' + str(args.margin) + ')' if args.loss in ('arcface', 'cosface') else ''}",
        flush=True,
    )
    if args.resume is not None:
        print(f"fine-tuning from: {args.resume} | prior_best_val_precision={best_score:.4f}", flush=True)
    if args.resume is not None:
        print("backbone initialization: loaded from resume checkpoint", flush=True)
    elif not args.pretrained:
        print("warning: ResNet18 is randomly initialized. Use --pretrained for transfer learning when weights are available.", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        matcher.train()
        running_loss, samples = 0.0, 0
        epoch_start = time.perf_counter()
        batches = len(loaders["train"])
        print(f"epoch {epoch}/{args.epochs} started ({batches} batches)", flush=True)
        for batch_index, batch in enumerate(loaders["train"], start=1):
            optimizer.zero_grad(set_to_none=True)
            first = encoder(batch["first_images"].to(device), batch["first_quality"].to(device), batch["first_valid"].to(device))[0]
            second = encoder(batch["second_images"].to(device), batch["second_quality"].to(device), batch["second_valid"].to(device))[0]
            labels = batch["label"].to(device)
            logits = matcher(first, second)
            # 主损失：配对二分类 BCE（可选 focal）；辅助损失按 --loss 计算
            primary = focal_bce_loss(logits, labels, args.focal_gamma) if args.focal_bce else criterion(logits, labels)
            auxiliary = compute_auxiliary_loss(args, aux_loss, first, second, batch["first_class"], batch["second_class"], labels, device)
            loss = primary + args.loss_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(matcher.parameters()), 1.0)
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            samples += labels.numel()
            if writer is not None:
                # 每 batch 记录一次训练损失，便于观察平滑的训练曲线
                writer.add_scalar("train/loss", loss.item(), (epoch - 1) * batches + batch_index)
            if batch_index % args.log_every == 0 or batch_index == batches:
                elapsed = time.perf_counter() - epoch_start
                batches_per_second = batch_index / max(elapsed, 1e-6)
                gpu_memory = ""
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    gpu_memory = f" | gpu_mem={allocated:.2f}/{reserved:.2f} GiB"
                print(
                    f"epoch {epoch}/{args.epochs} batch {batch_index}/{batches} | "
                    f"loss={running_loss / samples:.5f} | lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"speed={batches_per_second:.2f} batch/s{gpu_memory}",
                    flush=True,
                )
        validation_metrics = evaluate(encoder, matcher, loaders["validation"], device)
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(samples, 1),
            "epoch_seconds": time.perf_counter() - epoch_start,
            **validation_metrics,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if writer is not None:
            # 每轮记录验证指标与学习率，供 TensorBoard 展示
            writer.add_scalar("val/loss", validation_metrics["loss"], epoch)
            writer.add_scalar("val/accuracy", validation_metrics["accuracy"], epoch)
            writer.add_scalar("val/precision", validation_metrics["precision"], epoch)
            writer.add_scalar("val/recall", validation_metrics["recall"], epoch)
            writer.add_scalar("train/loss_epoch", row["train_loss"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        print(f"validation: {json.dumps(row)}", flush=True)
        if validation_metrics["precision"] > best_score:
            best_score = validation_metrics["precision"]
            torch.save(
                {"encoder": encoder.state_dict(), "matcher": matcher.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None, "args": vars(args), "epoch": epoch,
                 "aux_loss": aux_loss.state_dict() if aux_loss is not None else None},
                args.output_dir / "best.pt",
            )
            print(f"saved new best checkpoint: {args.output_dir / 'best.pt'} (val_precision={best_score:.4f})", flush=True)
        torch.save(
            {"encoder": encoder.state_dict(), "matcher": matcher.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler is not None else None, "args": vars(args), "epoch": epoch,
             "aux_loss": aux_loss.state_dict() if aux_loss is not None else None},
            args.output_dir / "last.pt",
        )

    best_path = args.output_dir / "best.pt"
    if not best_path.is_file():
        # Fine-tuning may not surpass the previous run's precision. In that case
        # evaluate the latest model and preserve the fact it did not improve.
        best_path = args.output_dir / "last.pt"
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    matcher.load_state_dict(checkpoint["matcher"])
    test_metrics = evaluate(encoder, matcher, loaders["test"], device)
    result = {
        "scope": "global inventory identity" if args.split_by == "global_id" else "sequence-local identity only",
        "resume_checkpoint": str(args.resume) if args.resume is not None else None,
        "best_checkpoint": str(best_path),
        "history": history,
        "test": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test: {json.dumps(result['test'])}", flush=True)
    print(f"saved metrics: {args.output_dir / 'metrics.json'}", flush=True)
    if writer is not None:
        # 最终测试集指标单独记录，随后关闭日志写入器
        for key, value in test_metrics.items():
            writer.add_scalar(f"test/{key}", value, args.epochs + 1)
        writer.close()


if __name__ == "__main__":
    main()
