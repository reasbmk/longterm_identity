"""Build short tracklets from MOT-format annotations.

The project dataset has per-sequence local identities. This script preserves
that scope in `identity_key` so code cannot silently treat IDs from different
sequences as the same physical object.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_sequence_size(sequence_dir: Path) -> tuple[int, int]:
    info = configparser.ConfigParser()
    info.read(sequence_dir / "seqinfo.ini", encoding="utf-8")
    section = info["Sequence"]
    return int(section["imWidth"]), int(section["imHeight"])


def parse_mot_rows(annotation_file: Path) -> Iterable[tuple[int, int, float, float, float, float, float, int, float]]:
    with annotation_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = [part.strip() for part in line.strip().split(",")]
            if not parts or not parts[0]:
                continue
            if len(parts) < 8:
                raise ValueError(f"{annotation_file}:{line_number} has fewer than 8 MOT columns")
            frame, track_id = int(parts[0]), int(parts[1])
            x, y, width, height = map(float, parts[2:6])
            confidence, class_id = float(parts[6]), int(float(parts[7]))
            visibility = float(parts[8]) if len(parts) > 8 else 1.0
            yield frame, track_id, x, y, width, height, confidence, class_id, visibility


def load_global_id_map(csv_path: Path | None) -> dict[tuple[str, int, int], list[tuple[int, int, str]]]:
    """读取 global_inventory_id CSV，返回 {(sequence, class, local): [(first, last, gid), ...]}。

    CSV 列为 sequence_id,class_id,local_track_id,global_inventory_id,first_frame,last_frame。
    一个 local track 可能被拆成多段（串号），每段一个全局编号 + 帧区间。
    gid 为 "invalid" 表示该段是假标注（如地面），应整帧丢弃。
    """
    if csv_path is None:
        return {}
    mapping: dict[tuple[str, int, int], list[tuple[int, int, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # 跳过表头
        for row in reader:
            if not row:
                continue
            mapping[(row[0], int(row[1]), int(row[2]))].append((int(row[4]), int(row[5]), row[3]))
    return dict(mapping)


def global_id_for_frame(segments: list[tuple[int, int, str]], frame: int) -> str | None:
    """返回 frame 所属段落的全局编号；不在任何段落返回 None。"""
    for first, last, gid in segments:
        if first <= frame <= last:
            return gid
    return None


def make_tracklets(dataset_root: Path, chunk_size: int, stride: int, min_frames: int,
                   global_id_map: dict | None = None, classes: set[int] | None = None) -> list[dict]:
    global_id_map = global_id_map or {}
    sequences_root = dataset_root / "sequences"
    annotations_root = dataset_root / "annotations"
    if not sequences_root.is_dir() or not annotations_root.is_dir():
        raise FileNotFoundError("Expected 'sequences' and 'annotations' under --dataset-root")

    records: list[dict] = []
    for sequence_dir in sorted(path for path in sequences_root.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        annotation_file = annotations_root / f"{sequence}.txt"
        if not annotation_file.exists():
            print(f"Skipping {sequence}: no annotation file")
            continue
        image_width, image_height = read_sequence_size(sequence_dir)
        groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for frame, local_id, x, y, width, height, confidence, class_id, visibility in parse_mot_rows(annotation_file):
            if classes is not None and class_id not in classes:
                continue
            image_path = sequence_dir / f"{frame:07d}.jpg"
            if not image_path.exists() or confidence <= 0 or width <= 1 or height <= 1:
                continue
            segments = global_id_map.get((sequence, class_id, local_id))
            global_id = global_id_for_frame(segments, frame) if segments else None
            if global_id == "invalid":
                continue  # 假标注帧（如地面），不进入训练
            groups[(class_id, local_id)].append({
                "frame": frame,
                "image_path": str(image_path.resolve()),
                "bbox_xywh": [x, y, width, height],
                "confidence": confidence,
                "visibility": visibility,
                "area_ratio": (width * height) / (image_width * image_height),
                "global_inventory_id": global_id,
            })

        for (class_id, local_id), observations in groups.items():
            observations.sort(key=lambda item: item["frame"])
            if len(observations) < min_frames:
                continue
            starts = range(0, len(observations), stride)
            for start in starts:
                chunk = observations[start : start + chunk_size]
                if len(chunk) < min_frames:
                    continue
                # 该 chunk 的全局编号取观测帧众数：串号边界处的 chunk 可能混入
                # 少量邻接物体的帧，众数给出最稳的标签。
                gid_counts = Counter(o["global_inventory_id"] for o in chunk)
                chunk_gid = gid_counts.most_common(1)[0][0] if gid_counts else None
                tracklet_id = f"{sequence}_c{class_id}_id{local_id}_s{chunk[0]['frame']}"
                records.append({
                    "tracklet_id": tracklet_id,
                    "sequence_id": sequence,
                    "class_id": class_id,
                    "local_track_id": local_id,
                    "identity_key": f"{sequence}:c{class_id}:id{local_id}",
                    "global_inventory_id": chunk_gid,
                    "identity_scope": "global_inventory" if chunk_gid else "sequence_local",
                    "observations": chunk,
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create attention-matcher tracklets from MOT annotations.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--global-id-csv", type=Path, default=None,
                        help="global_inventory_id 映射 CSV，用于给 tracklet 打跨路线全局编号。")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="只构建这些类别（1..10）；省略则全部类别。")
    args = parser.parse_args()
    if args.chunk_size < args.min_frames or args.stride < 1:
        raise ValueError("chunk-size must be >= min-frames and stride must be positive")

    global_id_map = load_global_id_map(args.global_id_csv)
    classes = set(args.classes) if args.classes else None
    records = make_tracklets(args.dataset_root, args.chunk_size, args.stride, args.min_frames, global_id_map, classes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    scope = "global inventory" if global_id_map else "sequence-local"
    print(f"Wrote {len(records)} tracklets to {args.output} (identity scope: {scope})")
    if not global_id_map:
        print("Warning: no --global-id-csv; identities are sequence-local and not valid across routes.")


if __name__ == "__main__":
    main()
