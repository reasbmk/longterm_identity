"""把「主 CSV（一条 local track 一个编号）」与「拆分表」合并成最终 global_inventory_id 映射。

主 CSV（如 global_inventory_id2.csv）是标注工具导出的 4 列表：每条 local track 一行。
但 GT 里部分 local track 混了多件物体（串号），拆分表（class2_splits.csv）把这些
track 按帧拆成多行（6 列，带 first_frame/last_frame）。

本脚本产出统一 6 列格式：非拆分的 track 用整条轨迹的首末帧补齐帧范围，
拆分的 track 直接沿用拆分表的行。下游 build_tracklets.py 据此给每条 chunk-tracklet
打上正确的全局编号。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def group_by_identity(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["identity_key"]].append(record)
    return groups


def frame_range(chunks: list[dict]) -> tuple[int, int]:
    """合并同一 local track 所有 chunk 的观测，返回 (首帧, 末帧)。"""
    frames = {int(obs["frame"]) for chunk in chunks for obs in chunk["observations"]}
    return min(frames), max(frames)


def main() -> None:
    manifest = Path("artifacts/tracklets_cached_224.jsonl")
    main_csv = Path("artifacts/global_inventory_id2.csv")
    split_csv = Path("artifacts/class2_splits.csv")
    output = Path("artifacts/global_inventory_id2_final.csv")

    groups = group_by_identity(load_manifest(manifest))

    # 主 CSV：第一行是表头，跳过
    main_rows = load_csv(main_csv)[1:]
    # 拆分表：第一行是表头，跳过；key = (seq, class, local)
    split_rows = load_csv(split_csv)[1:]
    split_keys = {(r[0], r[1], r[2]) for r in split_rows}

    rows: list[list] = []
    warnings: list[str] = []
    for seq, class_id, local_id, gid in main_rows:
        key = (seq, class_id, local_id)
        if key in split_keys:
            continue  # 该 track 有拆分，用拆分表的行替代
        identity_key = f"{seq}:c{class_id}:id{local_id}"
        if identity_key not in groups:
            warnings.append(f"{seq}_c{class_id}_id{local_id}: 主 CSV 有，但清单里找不到")
            continue
        first, last = frame_range(groups[identity_key])
        rows.append([seq, class_id, local_id, gid, first, last])

    # 拆分行原样并入（6 列）
    rows.extend(split_rows)

    # 排序：按全局编号、序列、类别、local_id
    rows.sort(key=lambda r: (r[3], r[0], int(r[1]), int(r[2])))

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_id", "class_id", "local_track_id", "global_inventory_id", "first_frame", "last_frame"])
        writer.writerows(rows)

    n_split = len(split_rows)
    n_single = len(rows) - n_split
    print(f"Wrote {output}")
    print(f"  单编号 track: {n_single} 行")
    print(f"  拆分 track:   {n_split} 行")
    print(f"  合计:         {len(rows)} 行")
    if warnings:
        print("\n警告：")
        for warning in warnings:
            print(f"  ! {warning}")


if __name__ == "__main__":
    main()
