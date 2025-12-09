#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Enhanced ReferSAM Architecture Diagram
包含完整的模型流程和详细的增强损失函数模块
适用于论文主图（Figure 1）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, ConnectionPatch
from matplotlib.patheffects import withStroke
import numpy as np
import os

# 专业配色方案
COLORS = {
    'input': '#E3F2FD',      # 浅蓝色 - 输入
    'encoder': '#FFF3E0',    # 浅橙色 - 编码器
    'adapter': '#E8F5E9',    # 浅绿色 - 适配器
    'fusion': '#F3E5F5',     # 浅紫色 - 融合模块
    'decoder': '#FCE4EC',    # 浅粉色 - 解码器
    'output': '#FFF9C4',     # 浅黄色 - 输出
    'loss': '#FFEBEE',       # 浅红色 - 损失函数
    'text': '#E0F2F1',       # 浅青色 - 文本
    'enhanced': '#C5E1A5',   # 中绿色 - 增强模块
    'border': '#424242',     # 深灰色 - 边框
    'arrow': '#1976D2',      # 蓝色 - 箭头
    'highlight': '#D32F2F',  # 红色 - 高亮（增强模块边框）
    'loss_highlight': '#FF6B35',  # 橙红色 - 损失函数高亮
}

def setup_figure(width=20, height=14, dpi=300):
    """设置专业图表"""
    fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor('white')
    return fig, ax

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor=None, 
             fontsize=10, fontweight='normal', alpha=1.0, zorder=2, 
             linewidth=1.5, text_color='black'):
    """绘制专业文本框"""
    if edgecolor is None:
        edgecolor = COLORS['border']
    
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle='round,pad=0.1',
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=linewidth,
                         alpha=alpha,
                         zorder=zorder)
    ax.add_patch(box)
    
    # 添加文字（带描边以提高可见性）
    txt = ax.text(x + width/2, y + height/2, text,
                  ha='center', va='center',
                  fontsize=fontsize,
                  fontweight=fontweight,
                  color=text_color,
                  zorder=zorder+1)
    if fontweight == 'bold':
        txt.set_path_effects([withStroke(linewidth=2, foreground='white', alpha=0.6)])
    
    return box

def draw_arrow(ax, start, end, color=None, style='solid', linewidth=2, 
               zorder=1, arrowstyle='->', mutation_scale=20):
    """绘制箭头"""
    if color is None:
        color = COLORS['arrow']
    
    linestyle = '-' if style == 'solid' else '--'
    arrow = FancyArrowPatch(start, end,
                           arrowstyle=arrowstyle,
                           mutation_scale=mutation_scale,
                           linewidth=linewidth,
                           color=color,
                           linestyle=linestyle,
                           zorder=zorder)
    ax.add_patch(arrow)
    return arrow

def draw_comprehensive_architecture():
    """绘制完整的总体架构图"""
    fig, ax = setup_figure(20, 14)
    
    # 标题
    ax.text(10, 13.2, 'Enhanced ReferSAM: Overall Architecture', 
            ha='center', va='center', 
            fontsize=24, fontweight='bold', 
            family='sans-serif')
    
    # ========== 1. 输入层 ==========
    img_box = draw_box(ax, 0.5, 11, 2.2, 1.3, 'Image\n[B×3×H×W]',
                      COLORS['input'], fontsize=11, fontweight='bold')
    text_box = draw_box(ax, 0.5, 9.2, 2.2, 1.3, 'Text Query\n[B×N_l]',
                       COLORS['text'], fontsize=11, fontweight='bold')
    
    # ========== 2. 编码器层 ==========
    # ViT图像编码器
    vit_encoder = draw_box(ax, 3.2, 10.5, 2.8, 2.2, 'ViT Image\nEncoder\n(SAM)',
                          COLORS['encoder'], fontsize=11, fontweight='bold')
    
    # 显示中间特征（小框）
    c1_label = ax.text(3.2, 9.8, 'C1 (1/4)', ha='left', va='center',
                       fontsize=8, style='italic', color='gray')
    c2_label = ax.text(3.2, 9.4, 'C2 (1/8)', ha='left', va='center',
                       fontsize=8, style='italic', color='gray')
    c3_label = ax.text(3.2, 9.0, 'C3 (1/16)', ha='left', va='center',
                       fontsize=8, style='italic', color='gray')
    c4_label = ax.text(3.2, 8.6, 'C4 (1/32)', ha='left', va='center',
                       fontsize=8, style='italic', color='gray')
    
    # 文本编码器
    text_encoder = draw_box(ax, 3.2, 7.5, 2.8, 1.3, 'BERT Text\nEncoder',
                           COLORS['encoder'], fontsize=11, fontweight='bold')
    
    # ========== 3. ViT适配器增强模块 ==========
    # 适配器主框架（大框）
    adapter_frame = FancyBboxPatch((6.5, 7), 6.5, 4.5,
                                   boxstyle='round,pad=0.2',
                                   facecolor=COLORS['adapter'],
                                   edgecolor=COLORS['border'],
                                   linewidth=2,
                                   alpha=0.15,
                                   zorder=0)
    ax.add_patch(adapter_frame)
    
    # 适配器标题
    ax.text(9.75, 11.2, 'ViT Adapter (Enhanced Modules)', 
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor=COLORS['highlight'],
                     linewidth=2))
    
    # 3.1 文本注意力聚合模块
    text_attn_frame = FancyBboxPatch((6.8, 9.8), 2.8, 1.2,
                                     boxstyle='round,pad=0.1',
                                     facecolor=COLORS['enhanced'],
                                     edgecolor=COLORS['highlight'],
                                     linewidth=2.5,
                                     alpha=0.7,
                                     zorder=1)
    ax.add_patch(text_attn_frame)
    text_attn_box = draw_box(ax, 6.8, 9.8, 2.8, 1.2, 
                            'Text Attention\nAggregation',
                            COLORS['enhanced'],
                            edgecolor=COLORS['highlight'],
                            fontsize=10, fontweight='bold',
                            linewidth=2.5)
    
    # 3.2 多尺度融合模块
    msf_frame = FancyBboxPatch((10.2, 9.8), 2.8, 1.2,
                               boxstyle='round,pad=0.1',
                               facecolor=COLORS['enhanced'],
                               edgecolor=COLORS['highlight'],
                               linewidth=2.5,
                               alpha=0.7,
                               zorder=1)
    ax.add_patch(msf_frame)
    msf_box = draw_box(ax, 10.2, 9.8, 2.8, 1.2,
                      'MultiScale Fusion\n(C2, C3, C4)',
                      COLORS['enhanced'],
                      edgecolor=COLORS['highlight'],
                      fontsize=10, fontweight='bold',
                      linewidth=2.5)
    
    # 3.3 增强C1C2融合模块
    c1c2_frame = FancyBboxPatch((6.8, 8.2), 2.8, 1.2,
                                boxstyle='round,pad=0.1',
                                facecolor=COLORS['enhanced'],
                                edgecolor=COLORS['highlight'],
                                linewidth=2.5,
                                alpha=0.7,
                                zorder=1)
    ax.add_patch(c1c2_frame)
    c1c2_box = draw_box(ax, 6.8, 8.2, 2.8, 1.2,
                       'Enhanced C1C2\nFusion',
                       COLORS['enhanced'],
                       edgecolor=COLORS['highlight'],
                       fontsize=10, fontweight='bold',
                       linewidth=2.5)
    
    # 特征输出标注
    adapter_feat_label = ax.text(10.2, 8.5, 'Adapter Features\n[B×C×H×W]',
                                 ha='center', va='center',
                                 fontsize=9, style='italic',
                                 bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='white',
                                          edgecolor=COLORS['border'],
                                          linewidth=1))
    
    # ========== 4. 特征生成层 ==========
    mask_feat_box = draw_box(ax, 13.8, 9.5, 2.5, 1.5, 
                            'Mask Feature\nGeneration',
                            COLORS['decoder'], fontsize=10, fontweight='bold')
    
    # ========== 5. 解码器层 ==========
    decoder_box = draw_box(ax, 13.8, 7.5, 2.5, 1.5, 
                          'SAM Mask\nDecoder',
                          COLORS['decoder'], fontsize=11, fontweight='bold')
    
    # ========== 6. 输出层 ==========
    output_box = draw_box(ax, 13.8, 5.5, 2.5, 1.3, 
                         'Predicted Mask\n[B×H×W]',
                         COLORS['output'], fontsize=11, fontweight='bold')
    
    # ========== 7. 增强损失函数模块（详细版）==========
    # 损失函数主框架
    loss_frame = FancyBboxPatch((0.5, 0.5), 13, 4.5,
                                boxstyle='round,pad=0.2',
                                facecolor=COLORS['loss'],
                                edgecolor=COLORS['loss_highlight'],
                                linewidth=3,
                                alpha=0.2,
                                zorder=0)
    ax.add_patch(loss_frame)
    
    # 损失函数标题
    ax.text(7, 4.7, 'Enhanced Loss Functions (Our Contribution)', 
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor=COLORS['loss_highlight'],
                     linewidth=2.5))
    
    # 输入（预测和GT）
    pred_input = draw_box(ax, 1, 3.4, 1.8, 0.9, 'Predicted\nMask',
                         COLORS['input'], fontsize=9, fontweight='bold')
    gt_input = draw_box(ax, 1, 2.2, 1.8, 0.9, 'Ground\nTruth',
                       COLORS['input'], fontsize=9, fontweight='bold')
    
    # 基础损失（标注）
    ax.text(3.2, 3.9, 'Base Losses', ha='center', va='center',
            fontsize=8, style='italic', color='gray')
    ce_box = draw_box(ax, 3.5, 3.5, 1.6, 0.7, 'CE Loss',
                     COLORS['loss'], fontsize=9)
    dice_box = draw_box(ax, 3.5, 2.7, 1.6, 0.7, 'Dice Loss',
                       COLORS['loss'], fontsize=9)
    
    # 增强损失（用高亮边框，标注）
    ax.text(6.1, 3.9, 'Enhanced Losses', ha='center', va='center',
            fontsize=8, style='italic', 
            color=COLORS['loss_highlight'], fontweight='bold')
    focal_box = draw_box(ax, 5.8, 3.5, 1.6, 0.7, 'Focal Loss\n(Imbalance)',
                        COLORS['enhanced'],
                        edgecolor=COLORS['loss_highlight'],
                        fontsize=9, fontweight='bold',
                        linewidth=2)
    iou_box = draw_box(ax, 5.8, 2.7, 1.6, 0.7, 'IoU Loss\n(Accuracy)',
                      COLORS['enhanced'],
                      edgecolor=COLORS['loss_highlight'],
                      fontsize=9, fontweight='bold',
                      linewidth=2)
    boundary_box = draw_box(ax, 5.8, 1.9, 1.6, 0.7, 'Boundary Loss\n(Edges)',
                           COLORS['enhanced'],
                           edgecolor=COLORS['loss_highlight'],
                           fontsize=9, fontweight='bold',
                           linewidth=2)
    
    # 自适应权重模块
    adaptive_box = draw_box(ax, 8.2, 2.4, 2.3, 1.8, 
                           'Adaptive\nWeighting\n(Learnable\nWeights)',
                           COLORS['fusion'], fontsize=10, fontweight='bold')
    
    # 课程学习模块
    curriculum_box = draw_box(ax, 11.2, 2.4, 2.3, 1.8,
                             'Curriculum\nLearning\n(Progressive)',
                             COLORS['adapter'], fontsize=10, fontweight='bold')
    
    # 课程学习时间线标注（更清晰）
    curriculum_timeline = ax.text(11.2, 1.0,
                                  'Epoch 0-2: Base Losses Only\n'
                                  'Epoch 2-8: + Focal Loss\n'
                                  'Epoch 8-15: + IoU Loss\n'
                                  'Epoch 15+: + Boundary Loss',
                                  ha='center', va='top',
                                  fontsize=8, style='italic',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white',
                                           edgecolor=COLORS['border'],
                                           linewidth=1.5))
    
    # 总损失输出
    total_loss_box = draw_box(ax, 14.5, 2.4, 1.5, 1.8, 
                              'Total\nLoss\n(Backward)',
                              COLORS['output'], fontsize=11, fontweight='bold')
    
    # ========== 箭头连接 ==========
    # 输入到编码器
    draw_arrow(ax, (2.7, 11.65), (3.2, 11.65))
    draw_arrow(ax, (2.7, 9.85), (3.2, 8.15))
    
    # 编码器到适配器
    draw_arrow(ax, (6.0, 11.6), (6.5, 10.5))
    draw_arrow(ax, (6.0, 11.2), (6.5, 10.2))
    draw_arrow(ax, (6.0, 10.8), (6.5, 9.9))
    draw_arrow(ax, (6.0, 10.4), (6.5, 9.6))
    draw_arrow(ax, (6.0, 8.15), (6.5, 10.4))
    
    # 适配器内部（简化表示）
    draw_arrow(ax, (9.6, 10.4), (10.2, 10.4))
    draw_arrow(ax, (9.6, 8.8), (10.2, 8.8))
    
    # 适配器到特征生成
    draw_arrow(ax, (13.3, 10.4), (13.8, 10.25))
    
    # 特征生成到解码器
    draw_arrow(ax, (14.95, 10.25), (14.95, 9.25))
    
    # 解码器到输出
    draw_arrow(ax, (14.95, 8.25), (14.95, 6.15))
    
    # 输出到损失函数（虚线，表示训练时）
    draw_arrow(ax, (13.8, 5.5), (2.8, 3.85),
              color=COLORS['loss_highlight'], style='dashed', linewidth=2.5)
    draw_arrow(ax, (13.8, 5.5), (2.8, 2.65),
              color=COLORS['loss_highlight'], style='dashed', linewidth=2.5)
    
    # 损失函数内部箭头
    # 输入到基础损失
    draw_arrow(ax, (2.8, 3.85), (3.5, 3.85))
    draw_arrow(ax, (2.8, 2.65), (3.5, 3.05))
    draw_arrow(ax, (2.8, 2.65), (3.5, 3.05))
    
    # 基础损失到增强损失
    draw_arrow(ax, (5.1, 3.85), (5.8, 3.85))
    draw_arrow(ax, (5.1, 3.05), (5.8, 3.05))
    draw_arrow(ax, (5.1, 2.25), (5.8, 2.25))
    
    # 所有损失到自适应权重
    draw_arrow(ax, (7.4, 3.85), (8.2, 3.3))
    draw_arrow(ax, (7.4, 3.05), (8.2, 3.3))
    draw_arrow(ax, (7.4, 2.25), (8.2, 3.0))
    
    # 自适应权重到课程学习
    draw_arrow(ax, (10.5, 3.3), (11.2, 3.3))
    
    # 课程学习到总损失
    draw_arrow(ax, (13.5, 3.3), (14.5, 3.3))
    
    # 标注：训练时
    ax.text(8.3, 5.0, 'Training Loss (Backward Propagation)', 
            ha='center', va='center',
            fontsize=10, style='italic', 
            color=COLORS['loss_highlight'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                     facecolor='white',
                     edgecolor=COLORS['loss_highlight'],
                     linewidth=2))
    
    # ========== 图例 ==========
    legend_x = 17
    legend_y = 11
    
    # 增强模块图例
    enhanced_legend = FancyBboxPatch((legend_x, legend_y), 2.5, 0.8,
                                     boxstyle='round,pad=0.1',
                                     facecolor=COLORS['enhanced'],
                                     edgecolor=COLORS['highlight'],
                                     linewidth=2.5)
    ax.add_patch(enhanced_legend)
    ax.text(legend_x + 1.25, legend_y + 0.4, 'Enhanced Module',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 损失函数图例
    loss_legend = FancyBboxPatch((legend_x, legend_y - 1.2), 2.5, 0.8,
                                 boxstyle='round,pad=0.1',
                                 facecolor=COLORS['enhanced'],
                                 edgecolor=COLORS['loss_highlight'],
                                 linewidth=2.5)
    ax.add_patch(loss_legend)
    ax.text(legend_x + 1.25, legend_y - 0.8, 'Enhanced Loss',
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 保存
    output_path = 'draw/img/comprehensive_architecture.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive architecture saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 70)
    print("Drawing Comprehensive Enhanced ReferSAM Architecture")
    print("Includes detailed Enhanced Loss Functions module")
    print("=" * 70)
    
    draw_comprehensive_architecture()
    
    print("\n" + "=" * 70)
    print("Diagram saved to draw/img/comprehensive_architecture.png")
    print("Ready for paper submission!")
    print("=" * 70)

