# 模型推理可视化工具

这个工具用于可视化指代表达式语义分割模型的推理结果，将原图和推理后的mask叠加显示在一起。支持可选的bounding box显示。

## 功能特点

- 支持从数据集批量可视化
- 支持单张图片可视化
- 原图和推理mask叠加显示（默认）
- 可选的bounding box显示（需使用--show_bbox）
- 支持显示ground truth对比

## 使用方法

### 1. 从数据集可视化（推荐）

```bash
# 可视化数据集中的5个随机样本
python visual/visualize_inference.py --mode dataset --num_samples 5 --dataset_name refcoco --split val

# 可视化指定索引的样本
python visual/visualize_inference.py --mode dataset --dataset_name refcoco --split val --sample_idx 10

# 不显示ground truth
python visual/visualize_inference.py --mode dataset --num_samples 5 --dataset_name refcoco --split val --no-show_gt

# 显示bounding box（默认不显示，语义分割任务通常不需要）
python visual/visualize_inference.py --mode dataset --num_samples 5 --dataset_name refcoco --split val --show_bbox
```

### 2. 单张图片可视化

```bash
python visual/visualize_inference.py --mode single --image_path /path/to/image.jpg --sentence "a red car" --output_dir visual/results
```

## 参数说明

### 通用参数
- `--mode`: 可视化模式，`dataset` 或 `single`
- `--output_dir`: 输出目录，默认为 `visual/results`

### 数据集模式参数
- `--num_samples`: 要可视化的样本数量（默认5）
- `--dataset_name`: 数据集名称，可选：`refcoco`, `refcoco+`, `refcocog`, `grefcoco`, `referit`
- `--split`: 数据集split，如 `val`, `test`, `train`
- `--sample_idx`: 指定数据集的样本索引（可选）
- `--show_gt`: 是否显示ground truth（默认True）
- `--show_bbox`: 是否显示bounding box（默认False，语义分割任务通常不需要）

### 单张图片模式参数
- `--image_path`: 图片路径
- `--sentence`: 文本描述
- `--show_bbox`: 是否显示bounding box（默认False）

## 输出说明

可视化结果会保存为PNG图片，包含：
1. **原始图像**：未处理的原始图像
2. **推理结果叠加**：原图 + 预测mask（红色半透明）+ 可选的预测bounding box（红色框，需使用--show_bbox）
3. **Ground Truth叠加**（如果启用）：原图 + GT mask（绿色半透明）+ 可选的GT bounding box（绿色框，需使用--show_bbox）

注意：这是指代表达式语义分割任务，默认只显示mask叠加，不显示bounding box。如需显示框，请使用 `--show_bbox` 参数。

## 示例

```bash
# 可视化refcoco验证集的10个样本
python visual/visualize_inference.py \
    --mode dataset \
    --num_samples 10 \
    --dataset_name refcoco \
    --split val \
    --output_dir visual/refcoco_val_results

# 可视化单张图片
python visual/visualize_inference.py \
    --mode single \
    --image_path /path/to/image.jpg \
    --sentence "a person riding a bicycle" \
    --output_dir visual/single_results
```

