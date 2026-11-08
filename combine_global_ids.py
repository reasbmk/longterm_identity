"""把各分类的 global_inventory_id{N}.csv 合成一份统一映射。

build_tracklets.py 的 --global-id-csv 只接收单个文件，而标注是按类别分文件导出的
（global_inventory_id2.csv .. global_inventory_id10.csv）。本脚本把它们按 6 列格式
拼接成一份 global_inventory_id_all.csv，供下游一次性重建带全局编号的 tracklet。
"""
from __future__ import annotations

import csv
from pathlib import Path

HEADER = ["sequence_id", "class_id", "local_track_id", "global_inventory_id", "first_frame", "last_frame"]


def main() -> None:
    artifacts = Path("artifacts")
    # 合并全部已标注类别（1..10）
    classes = list(range(1, 11))
    rows: list[list[str]] = []
    for cls in classes:
        path = artifacts / f"global_inventory_id{cls}.csv"
        if not path.exists():
            print(f"跳过缺失: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or len(header) < 6:
                raise ValueError(f"{path} 不是 6 列格式，无法合并")
            rows.extend(r for r in reader if r)

    # 排序与标注工具导出保持一致：按全局编号、序列、类别、local_id
    rows.sort(key=lambda r: (r[3], r[0], int(r[1]), int(r[2])))

    out = artifacts / "global_inventory_id_all.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)

    # 简单统计：每个类别的行数
    from collections import Counter
    counts = Counter(r[1] for r in rows)
    print(f"Wrote {out} ({len(rows)} 行)")
    for cls in sorted(counts, key=int):
        print(f"  类别 {cls}: {counts[cls]} 行")


if __name__ == "__main__":
    main()
