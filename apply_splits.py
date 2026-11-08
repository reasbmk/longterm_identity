"""把人工记录的「串号拆分」文字描述转成 global_inventory_id CSV（带帧范围）。

背景：GT 标注里一个 local_id 有时混了多件不同物体（ID 切换）。人工在标注工具里
按采样出的 8 张裁剪图判断「前几张是 A、后几张是 B」，但工具只支持两段拆分，
这里用脚本把任意多段拆分（含 A-B-A）按帧号精确落到每帧。

拆分规则：为每张采样裁剪图指定全局编号，然后每个观测帧归给「帧号最近」的那张
采样图对应的编号，再把连续同编号的帧压成 first_frame/last_frame 区间。
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_by_identity(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["identity_key"]].append(record)
    return groups


def merge_observations(chunks: list[dict]) -> list[dict]:
    by_frame: dict[int, dict] = {}
    for chunk in chunks:
        for observation in chunk["observations"]:
            by_frame.setdefault(int(observation["frame"]), observation)
    return [by_frame[frame] for frame in sorted(by_frame)]


def sample_indices(total: int, count: int) -> list[int]:
    """与标注工具一致：从 [0,total) 均匀取 count 个下标。"""
    if total <= count:
        return list(range(total))
    step = (total - 1) / (count - 1)
    return sorted({round(i * step) for i in range(count)})


FRAMES_PER_TRACKLET = 8  # 与 annotate_global_id.py 默认一致

# 转写清单：(seq, class, local_id, [(起始crop, 结束crop, 全局编号), ...])
# crop 从 1 开始、闭区间；global_id 用 "c2_XX" 字符串，"invalid" 表示"地面/假标注"。
SPLITS = [
    ("001", 2, 11, [(1, 7, "c2_31"), (8, 8, "c2_20")]),      # 主编号=c2_31（来自 global_inventory_id2.csv）
    ("003", 2, 1, [(1, 4, "c2_75"), (5, 8, "c2_76")]),
    ("003", 2, 2, [(1, 4, "c2_76"), (5, 8, "c2_75")]),
    ("003", 2, 11, [(1, 5, "c2_65"), (6, 8, "c2_64")]),
    ("003", 2, 12, [(1, 4, "c2_64"), (5, 8, "c2_65")]),
    ("003", 2, 16, [(1, 4, "c2_60"), (5, 8, "c2_59")]),
    ("003", 2, 17, [(1, 5, "c2_59"), (6, 8, "c2_60")]),
    ("003", 2, 28, [(1, 4, "c2_77"), (5, 8, "c2_83")]),
    ("003", 2, 69, [(1, 6, "c2_27"), (7, 8, "c2_23")]),
    ("004", 2, 29, [(1, 4, "c2_19"), (5, 6, "c2_18")]),
    ("004", 2, 30, [(1, 4, "c2_18"), (5, 6, "c2_19")]),
    ("004", 2, 51, [(1, 7, "c2_70"), (8, 8, "c2_69")]),
    ("004", 2, 52, [(1, 7, "c2_69"), (8, 8, "c2_70")]),
    ("004", 2, 57, [(1, 5, "c2_40"), (6, 8, "c2_70")]),
    ("004", 2, 60, [(1, 6, "c2_66"), (7, 8, "c2_65")]),      # FLAG 原文"前6后3"=9>8，取"后2张"
    ("004", 2, 61, [(1, 6, "c2_65"), (7, 8, "c2_66")]),      # FLAG 同上
    ("004", 2, 71, [(1, 3, "c2_45"), (4, 5, "c2_47"), (6, 8, "c2_45")]),
    ("004", 2, 73, [(1, 3, "c2_47"), (4, 5, "c2_45"), (6, 8, "c2_47")]),
    ("004", 2, 76, [(1, 3, "c2_58"), (4, 6, "c2_54"), (7, 8, "c2_52")]),
    ("004", 2, 83, [(1, 3, "c2_54"), (4, 8, "c2_58")]),
    ("004", 2, 89, [(1, 6, "c2_52"), (7, 8, "c2_54")]),  # "62"系笔误，实为52（与CSV一致）
    ("004", 2, 90, [(1, 3, "c2_52"), (4, 4, "c2_53"), (5, 8, "c2_52")]),
    ("004", 2, 148, [(1, 2, "c2_71"), (3, 3, "c2_75")]),     # 原文"741"实为71（与CSV一致）
    ("005", 2, 35, [(1, 2, "c2_32"), (3, 8, "c2_34")]),
    ("005", 2, 45, [(1, 5, "c2_20"), (6, 8, "c2_15")]),
    ("006", 2, 498, [(1, 6, "c2_23"), (7, 8, "c2_27")]),
    ("006", 2, 503, [(1, 6, "c2_27"), (7, 8, "c2_23")]),
    ("006", 2, 519, [(1, 4, "c2_06"), (5, 8, "c2_08")]),
    ("008", 2, 6, [(1, 1, "c2_06"), (2, 5, "c2_03"), (6, 8, "c2_07")]),
    ("008", 2, 26, [(1, 1, "c2_21"), (2, 8, "c2_17")]),
    ("008", 2, 63, [(1, 1, "c2_63"), (2, 8, "c2_67")]),
    ("008", 2, 64, [(1, 1, "c2_67"), (2, 8, "c2_63")]),
    ("012", 2, 78, [(1, 1, "c2_75"), (2, 8, "c2_77")]),
    ("012", 2, 79, [(1, 1, "c2_74"), (2, 8, "c2_37")]),      # "其余37"=c2_37，与 CSV 一致
    ("012", 2, 253, [(1, 6, "c2_86"), (7, 8, "invalid")]),   # 主编号=c2_86（来自 CSV）；后两张地面→invalid
]


def crop_gids_from_segments(segments: list[tuple[int, int, str]], total_crops: int) -> list[str]:
    """把 (起始,结束,编号) 片段展开成每张裁剪图的编号列表。"""
    gids = [None] * total_crops
    for c1, c2, gid in segments:
        for i in range(c1 - 1, c2):  # crop 1-based → 0-based
            if 0 <= i < total_crops:
                gids[i] = gid
    return gids


def assign_frames(merged: list[dict], crop_frames: list[int], crop_gids: list[str]) -> list[list]:
    """每个观测帧归给帧号最近的那张采样图，再把连续同编号帧压成区间。"""
    labeled = []
    for observation in merged:
        frame = int(observation["frame"])
        nearest = min(range(len(crop_frames)), key=lambda i: abs(crop_frames[i] - frame))
        labeled.append((crop_gids[nearest], frame))
    segments: list[list] = []
    for gid, frame in labeled:
        if segments and segments[-1][0] == gid:
            segments[-1][2] = frame  # 延长 last_frame
        else:
            segments.append([gid, frame, frame])
    return segments


def main() -> None:
    manifest = Path("artifacts/tracklets_cached_224.jsonl")
    output = Path("artifacts/class2_splits.csv")
    groups = group_by_identity(load_manifest(manifest))

    rows: list[list] = []
    flags: list[str] = []
    for seq, class_id, local_id, segments in SPLITS:
        identity_key = f"{seq}:c{class_id}:id{local_id}"
        if identity_key not in groups:
            flags.append(f"{seq}_c{class_id}_id{local_id}: 清单里找不到")
            continue
        merged = merge_observations(groups[identity_key])
        indices = sample_indices(len(merged), FRAMES_PER_TRACKLET)
        crop_frames = [int(merged[i]["frame"]) for i in indices]
        total_crops = len(indices)
        crop_gids = crop_gids_from_segments(segments, total_crops)
        if None in crop_gids:
            flags.append(f"{seq}_c{class_id}_id{local_id}: 片段没覆盖全部 {total_crops} 张裁剪图")
            continue
        for gid, first_frame, last_frame in assign_frames(merged, crop_frames, crop_gids):
            rows.append([seq, class_id, local_id, gid, first_frame, last_frame])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_id", "class_id", "local_track_id", "global_inventory_id", "first_frame", "last_frame"])
        writer.writerows(rows)

    print(f"Wrote {output} ({len(rows)} rows)")
    print("分段结果（供核对）：")
    for row in rows:
        print(f"  {row[0]}_c{row[1]}_id{row[2]}: {row[3]} 帧 {row[4]}-{row[5]}")
    if flags:
        print("\n需要你确认的项：")
        for flag in flags:
            print(f"  ! {flag}")


if __name__ == "__main__":
    main()
