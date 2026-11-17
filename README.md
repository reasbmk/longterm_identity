# 长期库存身份匹配基线

本模块在 `Dataset/outdoor_mot_2025.12` 的本地 MOT 数据上训练基于注意力的短轨迹匹配器。它与目标检测器、BoT-SORT 解耦：输入为 MOT 格式轨迹，不依赖相机内参、畸变参数或三维地图。

## 模块功能

1. 将每条 `(序列、类别、本地轨迹 ID)` 的真实轨迹切分为短轨迹。
2. 从短轨迹中采样目标裁剪图。
3. 使用 ResNet-18 图像编码器和质量感知注意力池化，为每条短轨迹生成一个特征向量。
4. 学习判断两条同类别短轨迹是否属于同一个本地身份。

标注同时包含本地 ID 与人工链接后的 `global_inventory_id`。本地 ID 只在**同一序列、同一类别**内有效；`global_inventory_id` 则跨序列标识同一件真实物资。默认 `--split-by global_id` 会按全局物资编号划分 train/val/test，避免同一物资跨路线泄漏；仅当清单没有全局编号时才回退到 `--split-by sequence`（此时是序列内身份匹配基线，不能表述为跨路线去重）。

## 安装依赖

请使用与 GPU 兼容的 PyTorch、torchvision 环境。

```powershell
pip install -r longterm_identity/requirements.txt
```

## 构建短轨迹清单

在仓库根目录执行。此命令只读取数据集并写入元数据；训练时才会按需读取裁剪图。

```powershell
py -3.11 longterm_identity/build_tracklets.py `
  --dataset-root Dataset/outdoor_mot_2025.12 `
  --output longterm_identity/artifacts/tracklets.jsonl `
  --chunk-size 12 `
  --stride 8
```

输入标注采用十列 MOT 格式：

```text
frame,id,x,y,w,h,confidence,class,visibility,mark
```

其中真实轨迹的唯一键应为 `(sequence_id, class_id, local_id)`，不能只使用 `id`。不同类别可在同一帧重复使用相同的本地 ID。

## 预缓存目标裁剪图

默认清单指向完整 MOT 图片。下面的命令会将每个唯一目标观测裁剪为小 JPEG，并生成新的清单；`dataset.py` 在存在 `cached_crop_path` 时会自动使用缓存图。

请在没有其他训练任务读取同一数据集时运行，因为该过程会产生大量磁盘读写。

```powershell
py -3.11 longterm_identity/cache_crops.py `
  --manifest longterm_identity/artifacts/tracklets.jsonl `
  --output-manifest longterm_identity/artifacts/tracklets_cached_224.jsonl `
  --crop-root longterm_identity/artifacts/crops_224 `
  --image-size 224
```

之后训练时使用 `--manifest longterm_identity/artifacts/tracklets_cached_224.jsonl`。若只做快速预实验，可以生成 `160x160` 缓存图，并同时指定 `--image-size 160`。不要同时运行缓存任务和训练任务。

## 训练长期身份模型

默认使用序列 `008` 作为验证集、`012` 作为测试集，两者均不参与训练。若 torchvision 的 ImageNet 权重可下载或已缓存，可增加 `--pretrained`；否则可不使用该参数开始训练。

```powershell
py -3.11 longterm_identity/train.py `
  --manifest longterm_identity/artifacts/tracklets_cached_224.jsonl `
  --output-dir longterm_identity/artifacts/run1 `
  --epochs 30 `
  --frames-per-tracklet 8 `
  --batch-size 16 `
  --pretrained
```

`dataset.py` 会在图像进入 ResNet-18 前统一缩放，默认大小为 `224x224`。快速预实验可以使用 `--image-size 160` 或 `128`，但缓存图尺寸应与之对应。检查点会保存该设置，`deduplicate.py` 推理时会自动复用。

生成的 `best.pt` 包含图像编码器、注意力池化器和成对匹配器；`metrics.json` 记录的是短轨迹成对匹配指标，不是仓库最终库存数量的准确率。

### 关键训练选项

- `--loss`：辅助损失（主损失始终是配对二分类 BCE，辅助损失按 `0.2` 权重作用在嵌入上）。可选：
  `contrastive`（基线）、`infonce`、`focal_nce`（逐 query 焦点 InfoNCE，当前对比中 acc 最高）、`focal_pair_nce`（逐对焦点加权）、`arcface`、`cosface`、`center`、`none`。
- `--backbone`：编码器骨干。可选 `resnet18`（默认，512 维）、`resnet34`（512 维）、`resnet50`（2048 维，6 GB 显存需降低 batch）、`mobilenet_v3_small`（576 维，适合边缘部署）。`PairMatcher` 输入维随 `4 × feature_dim` 自动调整。
- `--cfsa`：启用跨帧稳定性注意力（见下）。
- `--split-by`：`global_id`（默认）或 `sequence`。
- `--pretrained`：使用 ImageNet 预训练权重。随机初始化 vs 迁移学习是重要的消融变量。

### 跨帧稳定性注意力（CFSA）

`--cfsa` 启用自研模块。它不再用全局平均池化，而是对轨迹 8 帧的空间特征图做两路加权后聚合：

- **通道判别性**：SE 式 MLP 学习哪些通道对区分不同物资更重要；
- **空间稳定性**：每个空间位置在相邻帧间的余弦一致性越高权重越大，让模型聚焦"跨帧都稳定出现"的物体部件，抑制单帧反光/遮挡噪声。

聚合出的每帧判别性向量仍走质量注意力做帧间加权。该模块针对相似物资的误识别设计，可作为 `--cfsa` 开/关的消融对照。

## 跟踪后的长期匹配与库存队列

真实推理时，应输入 YOLO 加 BoT-SORT 导出的 MOT 格式预测轨迹，不能把 GT 身份 ID 作为生产推理输入。长期匹配器仅比较同类别候选短轨迹，产生以下决策：

- `same_probability >= 0.95`：仅在全局一致性检查后才可合并。
- `same_probability <= 0.20`：建立新的库存候选。
- 其余情况：标记为 `pending_review`，等待人工复核。

在没有跨路线真实物资标注前，应将模型用于候选排序和人工复核，不应执行仓库范围的自动合并。

以下命令可单独给清单评分，并输出 CSV 审计队列。推理过程中不会读取 `identity_key`：

```powershell
py -3.11 longterm_identity/deduplicate.py `
  --manifest longterm_identity/artifacts/tracklets.jsonl `
  --checkpoint longterm_identity/artifacts/run1/best.pt `
  --output longterm_identity/artifacts/review_queue.csv
```

当前长期模型只使用序列内标签训练时，建议将 `--merge-threshold` 设为 `1.1` 来关闭自动合并，并利用产生的 `pending_review` 样本进行跨路线身份人工标注。

## 训练 YOLOv12 检测器

`YOLOv12-BoT-SORT-ReID-main` 提供检测器与 BoT-SORT 实现，但原始 `train.py` 仅针对单类别热红外无人机场景。仓储十类别数据应使用项目新增的 `train_warehouse.py` 和数据 YAML。

请在安装该仓库依赖后，从 `YOLOv12-BoT-SORT-ReID-main/BoT-SORT/yolov12` 目录执行。`--model` 应使用通用 YOLOv12 预训练权重；不要将仓库附带的 `MOT_yolov12n.pt` 作为仓储预训练基线，因为它来自不同的无人机任务。

下列参数是当前 RTX 4050 6 GB 显存上成功使用的配置；`imgsz=1280、batch=8` 会导致显存不足。

```powershell
py -3.11 train_warehouse.py `
  --model yolov12n.pt `
  --data ../../../longterm_identity/yolo_warehouse_data.yaml `
  --epochs 30 `
  --batch 4 `
  --imgsz 960 `
  --workers 4 `
  --run-test
```

输出的 `runs/warehouse/yolov12n_10class*/weights/best.pt` 是 BoT-SORT 的检测输入。BoT-SORT 本身是配置与评测对象，不单独训练。

Ultralytics 会在每个 epoch 写入 `results.csv`，其中包含训练/验证集的框回归、分类与 DFL 损失。自定义 `EPOCH_SUMMARY` 日志会将训练损失、验证集 Precision、Recall、mAP 和学习率输出为一行 JSON。`--run-test` 会在训练结束后对保留测试集执行一次评估，并将测试集 Precision、Recall、mAP 保存为 `test_metrics.json`；测试集不用于选择模型或调节阈值。

## 运行完整库存流水线

统一入口依次执行：仓储 YOLOv12n 检测、每类别独立的 BoT-SORT、长期身份匹配、库存决策队列生成。它读取 MOT 数据集中的无人机序列图片，推理时不读取任何 GT 身份 ID。

在 YOLOv12 目录执行：

```powershell
cd YOLOv12-BoT-SORT-ReID-main/BoT-SORT/yolov12
py -3.11 run_warehouse_inventory.py `
  --weights runs/warehouse/yolov12n_10class6/weights/best.pt `
  --dataset-root ../../../Dataset/outdoor_mot_2025.12 `
  --longterm-checkpoint ../../../longterm_identity/artifacts/run_finetune/best.pt `
  --output-dir ../../../longterm_identity/artifacts/inventory_pipeline/yolov12n_botsort `
  --imgsz 960 --conf 0.10 --iou 0.70 `
  --merge-threshold 1.1 --new-threshold 0.20
```

BoT-SORT 额外依赖线性分配库 `lap`，该依赖已加入本地 `requirements.txt`。若环境此前未安装，可执行一次：

```powershell
py -3.11 -m pip install --user "lap>=0.5.12"
```

若只想快速验证一条路线，可附加 `--sequences 001`。输出包括：

- `mot_tracks/<sequence>.txt`：检测加 BoT-SORT 的 MOT 格式预测轨迹。
- `predicted_tracklets.jsonl`：每个 `(序列、类别、BoT-SORT ID)` 一条候选短轨迹。
- `inventory_review_queue.csv`：`merged`、`new_candidate` 或 `pending_review` 决策记录。
- `pipeline_summary.json`：本次运行使用的参数与统计数量。

`--merge-threshold 1.1` 会主动关闭跨路线自动合并，所有存在不确定性的候选均保留给人工复核。只有在使用人工链接的真实物资 ID 验证后，才应把阈值设为不大于 `1.0` 的值。

## 可视化完整流水线

可视化脚本会在原始无人机图片上绘制检测和追踪结果。绿色表示新的库存候选，橙色表示待人工确认，蓝色表示已接受的合并，灰色表示未达到长期匹配最小帧数的短轨迹。

```powershell
py -3.11 visualize_inventory_pipeline.py `
  --sequence-dir ../../../Dataset/outdoor_mot_2025.12/sequences/001 `
  --mot-tracks ../../../longterm_identity/artifacts/inventory_pipeline/smoke_001/mot_tracks/001.txt `
  --decisions ../../../longterm_identity/artifacts/inventory_pipeline/smoke_001/inventory_review_queue.csv `
  --output ../../../longterm_identity/artifacts/inventory_pipeline/smoke_001/sequence_001_inventory.mp4 `
  --fps 10 --with-map --map-history 60 --map-smoothing 0.08
```

增加 `--scale 0.5` 可输出更小的视频；增加 `--preview` 可在生成时显示本地预览窗口。

`--with-map` 会在右侧显示相对图像平面地图与最近轨迹尾线。它不使用相机内参、无人机姿态、GPS 或真实世界坐标重建；它仅用于对照目标 ID、观察短轨迹连续性与 ID 跳变。默认 `--map-smoothing 0.15` 会平滑右侧地图点的视觉移动，`0.08` 更慢，`1.0` 使用未平滑的逐帧图像坐标。左侧检测框与 BoT-SORT 轨迹不会被该可视化平滑改动。颜色代表决策状态，不代表已认证的真实物资身份。

## 序列标注工具

### 生成 class 2（large cable）

py -3.11 annotate_global_id.py --manifest artifacts/tracklets_cached_224.jsonl --classes 2 --output artifacts/annotate_class2.html

### 一次生成多个类别

py -3.11 annotate_global_id.py --manifest artifacts/tracklets_cached_224.jsonl --classes 1 2 4 --output artifacts/annotate_c124.html

### 只生成某个类别 + 某几个序列

py -3.11 annotate_global_id.py --manifest artifacts/tracklets_cached_224.jsonl --classes 2 --sequences 001 002 --output artifacts/annotate_c2_001_002.html

### 裁剪图 200px、整帧图 800px

py -3.11 annotate_global_id.py --manifest artifacts/tracklets_cached_224.jsonl --classes 3 --output artifacts/annotate_class3.html --crop-size 200 --context-size 800

### 裁剪图采样 12 张（更密一点）

py -3.11 annotate_global_id.py --manifest artifacts/tracklets_cached_224.jsonl --classes 3 --output artifacts/annotate_class3.html --frames-per-tracklet 12
