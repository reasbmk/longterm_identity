"""Render a browser contact sheet for assigning global_inventory_id.

关键设计：一个「local track」(sequence, class, local_id) 只显示一张卡片。
`build_tracklets.py` 会把一条长轨迹按 stride 切成多个互相重叠的 chunk-tracklet，
如果按 chunk 出卡片，同一件物体会出现多张卡片，人工逐张填编号容易前后不一致
（上一次 class 3 就因此出现 `003,3,2` 被标成两个编号的冲突）。

这里先把同一 identity_key 的所有 chunk 合并、按帧去重，再整条轨迹只出一张卡片：
填一次全局编号即作用于整条轨迹，导出时每个 (sequence,class,local_id) 只出一行。

卡片给出两类视觉线索，均从整条轨迹里均匀采样、放大展示：
  - 目标裁剪图：首/中/尾均匀采样 frames 张，每张带 1..N 序号，看物体外观
  - 整帧上下文图：首/中/尾 3 张，看目标在场景中的位置与邻居排布

支持「串号拆分」：GT 有时会把一个 local_id 重复用于两件不同物体。此时
在卡片里填「前段编号 + 从第几张起 + 后段编号」，导出会按帧拆成两行
（各带 first_frame/last_frame），把混在一段轨迹里的两件物体分开。

本工具只读数据集与清单，只写生成的 HTML；CSV 由浏览器端"导出 CSV"按钮下载。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def load_manifest(path: Path) -> list[dict]:
    """读取 JSONL 清单，返回所有 tracklet 记录。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_by_identity(records: list[dict]) -> dict[str, list[dict]]:
    """按 identity_key（即 sequence:class:local_id）把 chunk-tracklet 归组。

    build_tracklets.py 里 identity_key = f"{sequence}:c{class_id}:id{local_id}"，
    同一个 local track 会被切成多个 chunk，它们共享同一个 identity_key。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["identity_key"]].append(record)
    return groups


def merge_observations(chunks: list[dict]) -> list[dict]:
    """合并同一 local track 所有 chunk 的观测，按帧号去重并升序排列。

    相邻 chunk 会重叠（stride < chunk_size），同一帧可能出现在多个 chunk 里；
    去重只保留第一次出现，避免同帧重复。
    """
    by_frame: dict[int, dict] = {}
    for chunk in chunks:
        for observation in chunk["observations"]:
            by_frame.setdefault(int(observation["frame"]), observation)
    return [by_frame[frame] for frame in sorted(by_frame)]


def sample_indices(total: int, count: int) -> list[int]:
    """从 [0, total) 里均匀取 count 个下标，短轨迹取全部。"""
    if total <= count:
        return list(range(total))
    step = (total - 1) / (count - 1)
    return sorted({round(i * step) for i in range(count)})


def image_to_b64(image: Image.Image) -> str:
    """PIL 图像转 base64，直接内嵌进 HTML。"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_full(path: str) -> Image.Image:
    with Image.open(path) as source:
        return source.convert("RGB")


def crop_thumb(observation: dict, size: int) -> str:
    """目标裁剪缩略图：优先用已缓存的 224 裁剪图，否则从整帧现裁。"""
    cached = observation.get("cached_crop_path")
    if cached and Path(cached).exists():
        with Image.open(cached) as source:
            thumb = source.convert("RGB")
    else:
        image = load_full(observation["image_path"])
        x, y, w, h = observation["bbox_xywh"]
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(image.width, int(x + w)), min(image.height, int(y + h))
        thumb = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
    thumb = thumb.resize((size, size), Image.Resampling.BILINEAR)
    return image_to_b64(thumb)


def context_thumb(observation: dict, max_dim: int) -> str:
    """整帧上下文图：缩到 max_dim 内并画红框，展示目标在场景中的位置。"""
    image = load_full(observation["image_path"])
    scale = max_dim / max(image.width, image.height)
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    x, y, w, h = observation["bbox_xywh"]
    box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(255, 40, 40), width=max(2, round(max_dim / 300)))
    return image_to_b64(image)


def class_name(class_id: int) -> str:
    names = {
        1: "small cable", 2: "large cable", 3: "large distribution box",
        4: "rectangular transformer", 5: "triangular transformer",
        6: "small rectangular transformer", 7: "medium distribution box",
        8: "second medium distribution box", 9: "small distribution box",
        10: "nano distribution box",
    }
    return names.get(class_id, f"class {class_id}")


def render_card(chunks: list[dict], merged: list[dict],
                crop_size: int, context_size: int, frames: int) -> str:
    """渲染一条 local track 的卡片：采样裁剪图列 + 首/中/尾整帧上下文 + 编号/拆分输入。"""
    sequence = chunks[0]["sequence_id"]
    class_id = chunks[0]["class_id"]
    local_id = chunks[0]["local_track_id"]
    display_id = f"{sequence}_c{class_id}_id{local_id}"

    # 外观：整条轨迹均匀采样 frames 张裁剪图，每张带 1..N 序号（供拆分时按"第几张"指定）
    crop_idx = sample_indices(len(merged), frames)
    crops = "".join(
        f'<span class="cropwrap">'
        f'<img src="data:image/jpeg;base64,{crop_thumb(merged[i], crop_size)}" '
        f'title="frame {merged[i]["frame"]}" alt="crop">'
        f'<span class="idx">{i + 1}</span></span>'
        for i in crop_idx
    )

    # 位置：首/中/尾最多 3 张整帧图，展示不同视角下的场景排布
    ctx_idx = sample_indices(len(merged), 3)
    ctx_imgs = "".join(
        f'<img class="ctx" src="data:image/jpeg;base64,{context_thumb(merged[i], context_size)}" '
        f'title="full frame {merged[i]["frame"]}, box drawn" alt="context">'
        for i in ctx_idx
    )

    # 采样帧号列表：导出拆分时据此换算成真实帧范围
    frame_list = ",".join(str(merged[i]["frame"]) for i in crop_idx)

    first_frame = merged[0]["frame"]
    last_frame = merged[-1]["frame"]
    chunk_note = f" · {len(chunks)} chunk" if len(chunks) > 1 else ""

    return f"""
    <div class="card" data-tracklet="{display_id}" data-sequence="{sequence}"
         data-class="{class_id}" data-local="{local_id}" data-frames="{frame_list}">
      <div class="hdr">
        <b>{display_id}</b>
        <span class="meta">seq {sequence} · c{class_id} ({class_name(class_id)}) · local_id {local_id} · {len(merged)}f ({first_frame}-{last_frame}){chunk_note}</span>
      </div>
      <div class="crops">{crops}</div>
      <div class="ctx-row">{ctx_imgs}</div>
      <div class="assign">
        全局编号 <input class="gid" type="text" inputmode="numeric" placeholder="只填数字，如 54">
        <span class="split">或拆分：从第 <input class="split-at" type="number" min="1" max="{len(crop_idx)}" placeholder="{len(crop_idx)//2 + 1}"> 张起是 <input class="gid2" type="text" inputmode="numeric" placeholder="后段数字"></span>
      </div>
    </div>
    """


def render_html(groups: dict[str, list[dict]], crop_size: int, context_size: int, frames: int) -> str:
    """生成自包含 HTML：按 class→sequence→local_id 排序，每 local track 一张卡片。"""
    cards = []
    summary: dict[str, int] = defaultdict(int)
    for chunks in sorted(
        groups.values(),
        key=lambda chunks: (int(chunks[0]["class_id"]), chunks[0]["sequence_id"], int(chunks[0]["local_track_id"])),
    ):
        merged = merge_observations(chunks)
        cards.append(render_card(chunks, merged, crop_size, context_size, frames))
        summary[f"c{int(chunks[0]['class_id'])} seq{chunks[0]['sequence_id']}"] += 1

    summary_html = "<br>".join(f"{k}: {v} 条" for k, v in sorted(summary.items()))
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>global_inventory_id 标注</title>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 16px; background: #f5f5f5; }}
  .bar {{ position: sticky; top: 0; background: #fff; border: 1px solid #ccc; border-radius: 8px;
          padding: 12px; margin-bottom: 16px; display: flex; gap: 16px; align-items: center; }}
  .bar button {{ padding: 8px 16px; font-size: 15px; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(640px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }}
  .hdr b {{ font-family: monospace; }}
  .meta {{ color: #666; font-size: 12px; }}
  .crops {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }}
  .cropwrap {{ position: relative; }}
  .crops img {{ width: {crop_size}px; height: {crop_size}px; object-fit: cover; border: 1px solid #ccc; display: block; }}
  .idx {{ position: absolute; top: 2px; left: 2px; background: rgba(0,0,0,.65); color: #fff;
          font-size: 11px; line-height: 1; padding: 2px 4px; border-radius: 2px; }}
  .ctx-row {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .ctx {{ max-width: 49%; border: 1px solid #ccc; border-radius: 4px; }}
  .assign {{ margin-top: 8px; font-size: 13px; }}
  .assign input {{ margin-left: 6px; padding: 3px 6px; font-family: monospace; }}
  .assign .gid {{ width: 130px; }}
  .assign .gid2 {{ width: 130px; }}
  .assign .split-at {{ width: 52px; }}
  .split {{ color: #666; margin-left: 10px; }}
  .summary {{ color: #666; font-size: 12px; }}
</style></head>
<body>
  <div class="bar">
    <div class="summary">范围统计（local track 数）<br>{summary_html}</div>
    <button onclick="exportCsv()">导出 CSV</button>
    <span id="hint" style="color:#888;font-size:13px;">只填数字即可（导出时自动加 c类_ 前缀并两位补零）；同一件填同一个数；一条轨迹混两件时：填「前段数 + 从第几张起 + 后段数」；假标注/地面填 invalid</span>
  </div>
  <div class="grid">{''.join(cards)}</div>
<script>
function exportCsv() {{
  // 每张卡片 = 一条 local track。正常导出一行；拆分（填了 gid2 和 split-at）则拆成两行，
  // 各带 first_frame/last_frame，把混在一段轨迹里的两件物体按帧分开。
  const cards = document.querySelectorAll('.card');
  const rows = [];
  for (const card of cards) {{
    const seq = card.dataset.sequence;
    const cls = card.dataset.class;
    const local = card.dataset.local;
    const frames = card.dataset.frames.split(',').map(Number);
    const first = frames[0], last = frames[frames.length - 1];
    // "invalid" 保持裸字符串（假标注/地面），其余加类别前缀 + 两位补零
    const fmtGid = (raw) => (raw.toLowerCase() === 'invalid') ? 'invalid' : ('c' + cls + '_' + raw.padStart(2, '0'));
    const gidRaw = card.querySelector('input.gid').value.trim();
    if (!gidRaw) continue;
    const gid = fmtGid(gidRaw);
    const gid2Raw = card.querySelector('input.gid2').value.trim();
    const gid2 = gid2Raw ? fmtGid(gid2Raw) : '';
    const splitAt = parseInt(card.querySelector('input.split-at').value, 10);
    if (gid2 && splitAt >= 1 && splitAt <= frames.length) {{
      const splitFrame = frames[splitAt - 1];  // 第 splitAt 张采样图对应的真实帧号
      rows.push([seq, cls, local, gid, first, splitFrame - 1]);
      rows.push([seq, cls, local, gid2, splitFrame, last]);
    }} else {{
      rows.push([seq, cls, local, gid, first, last]);
    }}
  }}
  rows.sort((a,b) => a[3].localeCompare(b[3]) || a[0].localeCompare(b[0]) || Number(a[1])-Number(b[1]) || Number(a[2])-Number(b[2]));
  let csv = 'sequence_id,class_id,local_track_id,global_inventory_id,first_frame,last_frame\\n';
  for (const r of rows) csv += r.join(',') + '\\n';
  const blob = new Blob([csv], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'global_inventory_id.csv';
  a.click();
}}
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an annotation contact sheet (one card per local track).")
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/tracklets_cached_224.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/annotate_global_id.html"))
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="要标注的类别 id（1..10）。省略则全部。")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="要标注的序列 id。省略则全部。")
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="最多显示多少条 local track（先快速预览用）。")
    parser.add_argument("--crop-size", type=int, default=160,
                        help="裁剪图显示尺寸（像素）。")
    parser.add_argument("--context-size", type=int, default=640,
                        help="整帧上下文图最长边（像素）。")
    parser.add_argument("--frames-per-tracklet", type=int, default=8,
                        help="每条轨迹采样几张裁剪图。")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    if args.classes:
        records = [r for r in records if int(r["class_id"]) in args.classes]
    if args.sequences:
        records = [r for r in records if r["sequence_id"] in args.sequences]

    # 先按 identity_key 归组，再按 class/sequence/local_id 排序截断
    groups = group_by_identity(records)
    if args.max_tracks:
        ordered = sorted(
            groups.items(),
            key=lambda item: (int(item[1][0]["class_id"]), item[1][0]["sequence_id"], int(item[1][0]["local_track_id"])),
        )[: args.max_tracks]
        groups = dict(ordered)

    if not groups:
        raise SystemExit("No local tracks match the requested scope.")

    html = render_html(groups, args.crop_size, args.context_size, args.frames_per_tracklet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({len(groups)} local tracks)")
    print("Open it in a browser, type a shared 全局编号 on cards that are the same item, then 导出 CSV.")


if __name__ == "__main__":
    main()
