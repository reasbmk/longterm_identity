"""Score tracklets against an inventory prototype list.

This is an offline audit tool. It never reads the manifest's identity_key;
that key exists only to supervise training from MOT ground truth.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import load_manifest
from model import PairMatcher, TrackletEncoder


def encode_record(record: dict, encoder: TrackletEncoder, frames: int, image_size: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    all_observations = record["observations"]
    # Long tracks should represent their beginning, middle, and end rather than
    # only the first few adjacent frames.
    if len(all_observations) > frames:
        indices = torch.linspace(0, len(all_observations) - 1, frames).round().long().tolist()
        observations = [all_observations[index] for index in indices]
    else:
        observations = all_observations
    images, quality = [], []
    for observation in observations:
        x, y, width, height = observation["bbox_xywh"]
        cached_path = observation.get("cached_crop_path")
        with Image.open(cached_path or observation["image_path"]) as source:
            image = source.convert("RGB")
            if not cached_path:
                left, top = max(0, int(x)), max(0, int(y))
                right, bottom = min(image.width, int(x + width)), min(image.height, int(y + height))
                image = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
            images.append(transform(image))
        sharpness_proxy = min(1.0, max(0.0, (width * height) / 40000.0))
        quality.append([observation["confidence"], observation["visibility"], sharpness_proxy])
    valid = torch.zeros(frames, dtype=torch.bool)
    valid[:len(images)] = True
    while len(images) < frames:
        images.append(torch.zeros_like(images[0]))
        quality.append([0.0, 0.0, 0.0])
    with torch.no_grad():
        return encoder(
            torch.stack(images).unsqueeze(0).to(device),
            torch.tensor(quality, dtype=torch.float32).unsqueeze(0).to(device),
            valid.unsqueeze(0).to(device),
        )[0].squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a conservative long-term identity review queue.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-tracklet", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None, help="Crop size. Defaults to the training checkpoint setting.")
    parser.add_argument("--merge-threshold", type=float, default=0.95)
    parser.add_argument("--new-threshold", type=float, default=0.20)
    args = parser.parse_args()
    if not 0 <= args.new_threshold < args.merge_threshold:
        raise ValueError("Require 0 <= new-threshold < merge-threshold")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = args.image_size or int(checkpoint.get("args", {}).get("image_size", 224))
    backbone = checkpoint.get("args", {}).get("backbone", "resnet18")
    use_cfsa = checkpoint.get("args", {}).get("cfsa", False)
    encoder = TrackletEncoder(backbone=backbone, use_cfsa=use_cfsa).to(device)
    matcher = PairMatcher(embedding_dim=encoder.feature_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    matcher.load_state_dict(checkpoint["matcher"])
    encoder.eval()
    matcher.eval()

    # Each prototype is an average of previously accepted same-class tracklets.
    # Local GT IDs are intentionally ignored below.
    prototypes: dict[int, list[dict]] = {}
    decisions: list[dict] = []
    for record in sorted(load_manifest(args.manifest), key=lambda item: (item["sequence_id"], item["observations"][0]["frame"])):
        embedding = encode_record(record, encoder, args.frames_per_tracklet, image_size, device)
        class_id = int(record["class_id"])
        candidates = prototypes.get(class_id, [])
        best_probability, best_index = 0.0, None
        for index, candidate in enumerate(candidates):
            with torch.no_grad():
                probability = torch.sigmoid(matcher(embedding.unsqueeze(0), candidate["embedding"].unsqueeze(0))).item()
            if probability > best_probability:
                best_probability, best_index = probability, index

        inventory_id = ""
        if best_index is not None and best_probability >= args.merge_threshold:
            candidate = candidates[best_index]
            candidate["embedding"] = torch.nn.functional.normalize((candidate["embedding"] * candidate["count"] + embedding) / (candidate["count"] + 1), dim=0)
            candidate["count"] += 1
            inventory_id, decision = candidate["inventory_id"], "merged"
        elif best_probability <= args.new_threshold:
            inventory_id = f"candidate_c{class_id}_{len(candidates):05d}"
            prototypes.setdefault(class_id, []).append({"inventory_id": inventory_id, "embedding": embedding, "count": 1})
            decision = "new_candidate"
        else:
            decision = "pending_review"

        decisions.append({
            "tracklet_id": record["tracklet_id"],
            "sequence_id": record["sequence_id"],
            "class_id": class_id,
            "decision": decision,
            "inventory_candidate_id": inventory_id,
            "best_same_probability": f"{best_probability:.6f}",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]) if decisions else ["tracklet_id"])
        writer.writeheader()
        writer.writerows(decisions)
    print(f"Wrote {len(decisions)} decisions to {args.output}")
    print("Warning: only use cross-route merges after validating global physical-item labels.")


if __name__ == "__main__":
    main()
