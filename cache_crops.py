"""Cache MOT object crops and write a manifest that references those small images.

Overlapping tracklets contain repeated observations. Each unique source image and
box is cropped only once, then every repeated observation points to that file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def crop_key(sequence_id: str, local_track_id: int, frame: int, box: list[float]) -> str:
    raw = f"{sequence_id}|{local_track_id}|{frame}|" + ",".join(f"{value:.3f}" for value in box)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_crop(observation: dict, record: dict, output_root: Path, image_size: int, quality: int) -> Path:
    box = observation["bbox_xywh"]
    key = crop_key(record["sequence_id"], int(record["local_track_id"]), int(observation["frame"]), box)
    path = output_root / record["sequence_id"] / f"{key}.jpg"
    if path.exists():
        return path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(observation["image_path"]) as source:
        image = source.convert("RGB")
        x, y, width, height = box
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(image.width, int(x + width)), min(image.height, int(y + height))
        crop = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
        crop.resize((image_size, image_size), Image.Resampling.BILINEAR).save(path, "JPEG", quality=quality, optimize=True)
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-crop tracklet observations for faster training.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if args.image_size < 64:
        raise ValueError("image-size must be at least 64")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")

    records = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    created, reused = 0, 0
    seen: set[Path] = set()
    for record_index, record in enumerate(records, start=1):
        for observation in record["observations"]:
            box = observation["bbox_xywh"]
            key = crop_key(record["sequence_id"], int(record["local_track_id"]), int(observation["frame"]), box)
            expected_path = (args.crop_root / record["sequence_id"] / f"{key}.jpg").resolve()
            existed = expected_path.exists()
            observation["cached_crop_path"] = str(cache_crop(observation, record, args.crop_root, args.image_size, args.jpeg_quality))
            if expected_path not in seen:
                created += int(not existed)
                reused += int(existed)
                seen.add(expected_path)
        if record_index % 500 == 0 or record_index == len(records):
            print(f"processed {record_index}/{len(records)} tracklets | unique crops={len(seen)}", flush=True)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.output_manifest}")
    print(f"crops: created={created}, reused={reused}, total_unique={len(seen)}")


if __name__ == "__main__":
    main()
