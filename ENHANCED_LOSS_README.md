# Enhanced Loss Functions for ReferSAM

## 概述

本增强损失函数模块为ReferSAM模型添加了多种先进的损失函数，以提升指代表达式语义分割的性能。

## 新增损失函数

### 1. Focal Loss
- **目的**: 处理类别不平衡问题
- **适用场景**: 指代表达式分割中正样本(目标区域)远少于负样本(背景)
- **参数**:
  - `alpha`: 正样本权重 (默认: 0.25)
  - `gamma`: 聚焦参数 (默认: 2.0)

### 2. IoU Loss
- **目的**: 直接优化IoU指标
- **适用场景**: 提升分割精度，与评估指标一致
- **特点**: 与Dice Loss互补，提供不同的优化方向

### 3. Boundary Loss
- **目的**: 提升边缘质量
- **适用场景**: 需要精确边界的分割任务
- **参数**:
  - `kernel_size`: 边界检测核大小 (默认: 3)

### 4. 自适应权重调整
- **目的**: 自动平衡多个损失函数
- **特点**: 避免手动调参，自动学习最优权重组合

## 使用方法

### 1. 基本使用

```python
# 使用增强损失函数
python train_enhanced_loss.py --use_enhanced_loss --loss_config_path configs/enhanced_loss_config.yaml

# 使用原始损失函数
python train_enhanced_loss.py
```

### 2. 配置文件

编辑 `configs/enhanced_loss_config.yaml` 来调整损失函数参数:

```yaml
# 损失函数开关
use_focal: true          # 是否使用Focal Loss
use_iou: true            # 是否使用IoU Loss
use_boundary: true       # 是否使用边界损失
use_adaptive_weighting: true  # 是否使用自适应权重

# Focal Loss参数
focal_alpha: 0.25        # 正样本权重
focal_gamma: 2.0         # 聚焦参数

# 边界损失参数
boundary_kernel_size: 3  # 边界检测核大小
```

### 3. 在代码中使用

```python
from model.enhanced_builder import refersam_enhanced
from model.enhanced_criterion import EnhancedSegMaskLoss

# 创建增强模型
model = refersam_enhanced(
    pretrained=None, 
    args=args, 
    loss_config_path='configs/enhanced_loss_config.yaml'
)

# 或者直接创建损失函数
criterion = EnhancedSegMaskLoss(
    use_focal=True,
    use_iou=True,
    use_boundary=True,
    use_adaptive_weighting=True
)
```

## 文件结构

```
model/
├── enhanced_criterion.py      # 增强损失函数实现
├── enhanced_builder.py        # 增强模型构建器
├── criterion.py              # 原始损失函数
└── builder.py                # 原始模型构建器

configs/
├── enhanced_loss_config.yaml  # 增强损失配置
└── main_refersam_bert.yaml   # 原始配置

train_enhanced_loss.py        # 增强训练脚本
train_bert_multiGpu.py        # 原始训练脚本
```

## 性能提升

基于您的训练日志分析，当前模型性能：
- Epoch 1: mIoU=0.48, oIoU=0.47
- Epoch 4: mIoU=0.63, oIoU=0.60

预期增强损失函数能带来：
- **Focal Loss**: 提升难样本学习，改善类别不平衡
- **IoU Loss**: 直接优化IoU指标，提升分割精度
- **Boundary Loss**: 改善边缘质量，提升边界精度
- **自适应权重**: 自动平衡各损失函数，避免手动调参

## 兼容性

- ✅ 完全向后兼容原始代码
- ✅ 可以随时切换回原始损失函数
- ✅ 支持所有原始训练参数
- ✅ 支持DeepSpeed分布式训练

## 消融实验

可以通过以下参数进行消融实验：

```bash
# 只使用Focal Loss
python train_enhanced_loss.py --use_enhanced_loss --loss_ablation focal

# 只使用IoU Loss
python train_enhanced_loss.py --use_enhanced_loss --loss_ablation iou

# 只使用边界损失
python train_enhanced_loss.py --use_enhanced_loss --loss_ablation boundary

# 使用所有增强损失
python train_enhanced_loss.py --use_enhanced_loss --loss_ablation all
```

## 注意事项

1. **内存使用**: 增强损失函数会略微增加内存使用
2. **训练时间**: 计算复杂度略有增加，但影响很小
3. **超参数**: 建议从默认参数开始，根据验证结果调整
4. **调试**: 可以通过日志查看各损失函数的权重变化

## 故障排除

如果遇到问题，可以：
1. 检查配置文件格式是否正确
2. 确认所有依赖包已安装
3. 查看训练日志中的损失函数权重
4. 尝试关闭某些损失函数进行调试
