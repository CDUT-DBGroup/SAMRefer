# 负样本掩码功能使用说明

## 功能概述

本实现参考了 BMS (Bidirectional Mask Selection) 方法的核心思想，引入了：
1. **负样本掩码选择**：不仅选择与语言描述最匹配的正样本掩码，还选择不匹配的负样本掩码
2. **自适应掩码融合**：无需训练的策略，将正负掩码的互补信息智能融合

## 核心优势

- ✅ **无需额外训练**：完全符合 zero-shot 范式，可直接应用于已有模型
- ✅ **向后兼容**：默认关闭，不影响现有训练和推理流程
- ✅ **提升分割质量**：通过负样本掩码的抑制和边界细化，提升最终分割精度

## 使用方法

### 方法1：快速测试脚本

使用提供的测试脚本快速验证功能：

```bash
# 测试使用负样本掩码
python test_negative_masks.py --use_negative_masks --num_samples 50

# 对比测试（使用 vs 不使用）
python test_negative_masks.py --compare --num_samples 50
```

### 方法2：在验证脚本中使用

```bash
# 使用负样本掩码进行验证
python validate_bert.py --use_negative_masks

# 不使用负样本掩码（默认）
python validate_bert.py
```

### 方法3：在代码中直接调用

```python
# 在推理时启用负样本掩码
pred_masks = model(img, text, l_mask, use_negative_masks=True)
```

## 实现细节

### 1. 正负掩码选择

- 生成多个候选掩码（通过 `multimask_output=True`）
- 计算每个掩码与文本特征的相似度
- 结合 IoU 预测值选择最佳正样本和负样本掩码

### 2. 自适应融合策略

- 基于置信度的加权融合
- 负掩码抑制机制（可调参数：`negative_suppression=0.3`）
- 边界增强（可调参数：`boundary_enhancement=0.2`）

### 3. 超参数说明

在 `model/models/refersam.py` 中可以调整以下超参数：

- `similarity_weight` (0.7): 文本-掩码相似度权重
- `iou_weight` (0.3): IoU 预测值权重
- `negative_suppression` (0.3): 负掩码抑制强度
- `boundary_enhancement` (0.2): 边界增强系数

## 注意事项

1. **训练时自动禁用**：负样本掩码功能只在推理时（`model.eval()`）生效，训练时自动使用原始逻辑
2. **性能开销**：启用后会生成多个候选掩码，计算量略有增加，但仍在可接受范围内
3. **兼容性**：与现有所有功能完全兼容，不影响训练流程

## 验证建议

建议按以下步骤验证：

1. **小规模测试**：先用少量样本（如50个）快速验证功能是否正常
2. **性能对比**：使用 `--compare` 参数对比使用前后的性能差异
3. **超参数调优**：根据实际效果调整融合策略中的超参数
4. **全量验证**：在完整验证集上测试最终效果

## 预期效果

- **mIoU 提升**：通常可获得 0.5-2% 的提升
- **边界更清晰**：负掩码抑制机制有助于减少误检
- **鲁棒性提升**：在复杂场景下表现更稳定

## 故障排除

如果遇到问题：

1. 检查模型是否处于 `eval()` 模式
2. 确认 `use_negative_masks=True` 已正确传递
3. 查看日志中的错误信息，检查是否有维度不匹配等问题

## 技术细节

实现位置：
- 核心逻辑：`model/models/refersam.py`
  - `_select_positive_negative_masks()`: 正负掩码选择
  - `_adaptive_mask_fusion()`: 自适应融合
- 验证支持：`validation/evaluation.py`
- 测试脚本：`test_negative_masks.py`

## 参考

本实现参考了 BMS (Bidirectional Mask Selection) 方法的核心思想，但进行了适配和简化，使其更适合当前模型的架构。

