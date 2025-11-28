# 模型架构图说明文档

## 概述

本目录包含了Enhanced ReferSAM模型的完整架构图，包括总体架构图和各个优化模块的详细图。

## 生成的图表文件

### 1. overall_architecture.png
**总体模型架构图**

展示了Enhanced ReferSAM的完整流程：
- **输入层**: 图像和文本输入
- **编码器**: ViT图像编码器和文本编码器
- **ViT适配器（增强模块）**: 包含三个核心优化模块
  - MultiScale Fusion（多尺度融合）
  - Enhanced C1C2 Fusion（C1C2融合）
  - Text Attention（文本注意力聚合）
- **特征生成**: Adapter特征和Mask特征
- **解码器**: SAM Mask Decoder
- **输出**: 预测的Mask
- **损失函数**: 增强损失函数（训练时）

**用途**: 论文的主要架构图（Figure 1或Figure 2）

---

### 2. multiscale_fusion.png
**多尺度特征融合模块详细图**

展示了MultiScaleFusion模块的内部结构：
- **输入**: C2 (1/8 scale), C3 (1/16 scale), C4 (1/32 scale)
- **跨尺度注意力**: 每个尺度关注其他尺度的特征
- **边界保护门控**: 保护边界信息不被过度平滑
- **输出**: 增强后的多尺度特征

**核心创新点**:
- 跨尺度注意力机制实现多尺度特征交互
- 门控机制保护边界信息
- 残差连接避免过度修改

**用途**: 论文中MultiScale Fusion模块的详细说明图

---

### 3. c1c2_fusion.png
**增强C1C2融合模块详细图**

展示了EnhancedC1C2Fusion模块的内部结构：
- **输入**: C1 (1/4 scale) 和 C2 (1/8 scale)
- **上采样**: 将C2上采样到C1的尺寸
- **通道注意力**: 关注重要通道
- **空间注意力**: 关注重要空间位置
- **门控融合**: 控制融合强度
- **输出**: 融合后的C1特征

**核心创新点**:
- 使用空间注意力替代简单的上采样相加
- 避免序列注意力的内存爆炸问题
- 门控机制保护原始特征

**用途**: 论文中Enhanced C1C2 Fusion模块的详细说明图

---

### 4. text_attention.png
**文本注意力聚合模块详细图**

展示了文本注意力聚合机制的内部结构：
- **输入**: 文本特征序列 [B×N×D]
- **基础Token**: EOS/CLS token（原始方法）
- **多头注意力**: 使用可学习的query聚合文本特征
- **加权融合**: 可学习权重融合基础token和注意力结果
- **输出**: 聚合后的文本特征

**核心创新点**:
- 使用注意力机制替代简单的token选择
- 可学习的query更好地理解文本语义
- 支持变长文本处理

**用途**: 论文中Text Attention Aggregation模块的详细说明图

---

### 5. enhanced_loss.png
**增强损失函数模块详细图**

展示了Enhanced Loss Functions模块的内部结构：
- **输入**: 预测Mask和Ground Truth
- **基础损失**: CE Loss, Dice Loss
- **增强损失**: Focal Loss, IoU Loss, Boundary Loss（新增）
- **自适应权重**: 自动学习各损失函数的最优权重
- **课程学习**: 渐进式引入复杂损失函数
- **输出**: 总损失

**核心创新点**:
- 多损失函数协同优化
- 自适应权重调整机制
- 课程学习策略

**用途**: 论文中Enhanced Loss Functions模块的详细说明图

---

## 图表使用建议

### 论文中的使用方式

1. **总体架构图 (overall_architecture.png)**
   - 放在论文的Method部分开头
   - 作为Figure 1或Figure 2
   - 标题: "Overall Architecture of Enhanced ReferSAM"

2. **模块详细图**
   - 放在对应模块的详细描述部分
   - 可以作为子图（subfigure）或独立图
   - 建议顺序：
     - Figure 3(a): MultiScale Fusion
     - Figure 3(b): Enhanced C1C2 Fusion
     - Figure 3(c): Text Attention Aggregation
     - Figure 4: Enhanced Loss Functions

### 图表特点

- **高分辨率**: 所有图表均为300 DPI，适合论文印刷
- **清晰标注**: 包含模块名称、数据维度、关键操作
- **颜色编码**: 
  - 蓝色系：输入/文本
  - 黄色系：编码器
  - 绿色系：适配器/增强模块
  - 红色边框：新增/改进的模块
  - 紫色系：解码器
  - 黄色：输出

### 修改图表

如果需要修改图表，可以编辑 `draw_model_architecture.py` 脚本：

```python
# 修改颜色
COLORS = {
    'input': '#E8F4F8',  # 修改输入颜色
    # ...
}

# 修改尺寸
fig, ax = plt.subplots(1, 1, figsize=(16, 10))  # 修改图表尺寸

# 修改字体
ax.text(..., fontsize=16, fontweight='bold')  # 修改字体大小
```

然后重新运行：
```bash
python draw_model_architecture.py
```

---

## 技术细节

### 依赖库
- matplotlib
- numpy

### 字体支持
脚本已配置中文字体支持，如果显示有问题，可以：
1. 安装中文字体（如SimHei）
2. 或修改脚本中的字体设置

### 输出格式
- 格式: PNG
- 分辨率: 300 DPI
- 背景: 白色
- 适合: 论文、演示文稿、技术文档

---

## 注意事项

1. **论文使用**: 这些图表可以直接用于学术论文，但建议根据期刊/会议的要求调整尺寸和字体
2. **版权**: 图表基于您的模型代码生成，版权归您所有
3. **修改**: 可以根据论文审稿意见或展示需求修改图表样式
4. **备份**: 建议保留原始脚本，以便后续修改

---

## 联系与支持

如有问题或需要进一步定制图表，请参考：
- 模型代码: `model/vit_adapter/` 和 `model/enhanced_criterion.py`
- 改进分析: `MODEL_IMPROVEMENT_ANALYSIS.md`

