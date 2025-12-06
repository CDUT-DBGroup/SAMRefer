#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional model architecture diagrams for Enhanced ReferSAM
Designed in the style of top-tier computer vision conferences (CVPR, ICCV, NeurIPS)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.patheffects import withStroke
import numpy as np
import os

# Professional color scheme inspired by top-tier papers
COLORS = {
    'input': '#E3F2FD',      # Light blue
    'encoder': '#FFF3E0',    # Light orange
    'adapter': '#E8F5E9',    # Light green
    'fusion': '#F3E5F5',     # Light purple
    'decoder': '#FCE4EC',    # Light pink
    'output': '#FFF9C4',     # Light yellow
    'loss': '#FFEBEE',       # Light red
    'text': '#E0F2F1',       # Light teal
    'enhanced': '#C5E1A5',   # Medium green (for enhanced modules)
    'border': '#424242',      # Dark gray for borders
    'arrow': '#1976D2',       # Blue for arrows
    'highlight': '#D32F2F',  # Red for highlights
}

# Professional styling
BOX_STYLE = {
    'linewidth': 1.5,
    'edgecolor': COLORS['border'],
    'facecolor': None,
    'boxstyle': 'round,pad=0.05',
}

ARROW_STYLE = {
    'arrowstyle': '->',
    'mutation_scale': 25,
    'linewidth': 2,
    'color': COLORS['arrow'],
    'zorder': 1,
}

def setup_figure(width=16, height=10, dpi=300):
    """Setup professional figure with proper styling"""
    fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_facecolor('white')
    return fig, ax

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor=None, 
             fontsize=11, fontweight='normal', alpha=1.0, zorder=2):
    """Draw a professional box with text"""
    if edgecolor is None:
        edgecolor = COLORS['border']
    
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle='round,pad=0.1',
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=1.5,
                         alpha=alpha,
                         zorder=zorder)
    ax.add_patch(box)
    
    # Add text with stroke for better visibility
    txt = ax.text(x + width/2, y + height/2, text,
                  ha='center', va='center',
                  fontsize=fontsize,
                  fontweight=fontweight,
                  color='black',
                  zorder=zorder+1)
    txt.set_path_effects([withStroke(linewidth=3, foreground='white', alpha=0.7)])
    
    return box

def draw_arrow(ax, start, end, color=None, style='solid', linewidth=2, zorder=1):
    """Draw a professional arrow"""
    if color is None:
        color = COLORS['arrow']
    
    linestyle = '-' if style == 'solid' else '--'
    arrow = FancyArrowPatch(start, end,
                           arrowstyle='->',
                           mutation_scale=25,
                           linewidth=linewidth,
                           color=color,
                           linestyle=linestyle,
                           zorder=zorder)
    ax.add_patch(arrow)
    return arrow

def draw_overall_architecture():
    """Draw overall architecture in professional style"""
    fig, ax = setup_figure(18, 11)
    
    # Title
    ax.text(9, 10.5, 'Enhanced ReferSAM Architecture', 
            ha='center', va='center', 
            fontsize=22, fontweight='bold', 
            family='sans-serif')
    
    # Input layer
    img_box = draw_box(ax, 1, 8.5, 2.5, 1.2, 'Image\n[B×3×H×W]',
                      COLORS['input'], fontsize=10, fontweight='bold')
    text_box = draw_box(ax, 1, 6.8, 2.5, 1.2, 'Text\n[B×N]',
                       COLORS['text'], fontsize=10, fontweight='bold')
    
    # Encoders
    vit_encoder = draw_box(ax, 4.5, 8.5, 2.5, 1.2, 'ViT Image\nEncoder',
                          COLORS['encoder'], fontsize=10, fontweight='bold')
    text_encoder = draw_box(ax, 4.5, 6.8, 2.5, 1.2, 'Text\nEncoder',
                           COLORS['encoder'], fontsize=10, fontweight='bold')
    
    # ViT Adapter (main module)
    adapter_main = draw_box(ax, 8, 6, 4.5, 3.5,
                           '', COLORS['adapter'], 
                           edgecolor=COLORS['border'], alpha=0.3, zorder=0)
    
    # Title for adapter
    ax.text(10.25, 9.2, 'ViT Adapter (Enhanced)', 
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor=COLORS['border'],
                     linewidth=1.5))
    
    # Enhanced modules inside adapter
    msf_box = draw_box(ax, 8.5, 7.8, 3.5, 0.8, 'MultiScale Fusion',
                      COLORS['enhanced'], 
                      edgecolor=COLORS['highlight'], 
                      fontsize=9, fontweight='bold')
    
    c1c2_box = draw_box(ax, 8.5, 6.8, 3.5, 0.8, 'Enhanced C1C2 Fusion',
                       COLORS['enhanced'],
                       edgecolor=COLORS['highlight'],
                       fontsize=9, fontweight='bold')
    
    text_attn_box = draw_box(ax, 8.5, 5.8, 3.5, 0.8, 'Text Attention Aggregation',
                            COLORS['enhanced'],
                            edgecolor=COLORS['highlight'],
                            fontsize=9, fontweight='bold')
    
    # Features
    feat_box = draw_box(ax, 13.5, 7, 2.5, 1.2, 'Adapter\nFeatures',
                       COLORS['adapter'], fontsize=10, fontweight='bold')
    
    # Mask feature
    mask_feat_box = draw_box(ax, 13.5, 5.2, 2.5, 1.2, 'Mask Feature\n& Prompts',
                            COLORS['decoder'], fontsize=10, fontweight='bold')
    
    # Decoder
    decoder_box = draw_box(ax, 13.5, 3, 2.5, 1.5, 'SAM Mask\nDecoder',
                          COLORS['decoder'], fontsize=11, fontweight='bold')
    
    # Output
    output_box = draw_box(ax, 13.5, 0.8, 2.5, 1.5, 'Predicted\nMask',
                         COLORS['output'], fontsize=11, fontweight='bold')
    
    # Loss function (training)
    loss_box = draw_box(ax, 1, 3, 3, 2, 'Enhanced Loss\nFunctions\n(Focal+IoU+Boundary)',
                       COLORS['loss'],
                       edgecolor=COLORS['highlight'],
                       fontsize=10, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, (3.5, 9.1), (4.5, 9.1))
    draw_arrow(ax, (3.5, 7.4), (4.5, 7.4))
    draw_arrow(ax, (7, 9.1), (8, 8.5))
    draw_arrow(ax, (7, 7.4), (8, 7))
    draw_arrow(ax, (12.5, 7.6), (13.5, 7.6))
    draw_arrow(ax, (14.75, 7.6), (14.75, 6.4))
    draw_arrow(ax, (14.75, 5.2), (14.75, 4.5))
    draw_arrow(ax, (14.75, 3), (14.75, 2.3))
    
    # Loss arrow (dashed)
    draw_arrow(ax, (13.5, 1.55), (4, 4),
              color=COLORS['highlight'], style='dashed', linewidth=2)
    ax.text(8.5, 1.5, 'Training Loss', ha='center', va='center',
           fontsize=9, style='italic', color=COLORS['highlight'])
    
    # Save
    output_path = 'draw/img/overall_architecture.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Overall architecture saved to {output_path}")
    plt.close()


def draw_multiscale_fusion():
    """Draw MultiScale Fusion module"""
    fig, ax = setup_figure(14, 9)
    
    ax.text(7, 8.3, 'MultiScale Fusion Module', 
            ha='center', va='center',
            fontsize=18, fontweight='bold')
    
    # Input features
    y_positions = [6.5, 4.5, 2.5]
    scales = ['C2\n1/8 scale', 'C3\n1/16 scale', 'C4\n1/32 scale']
    
    for i, (y, scale) in enumerate(zip(y_positions, scales)):
        draw_box(ax, 1, y, 2, 1.2, scale,
                COLORS['input'], fontsize=10, fontweight='bold')
    
    # Cross-scale attention
    attn_box = draw_box(ax, 4, 2, 3.5, 5,
                       'Cross-Scale\nAttention\nMechanism',
                       COLORS['fusion'], fontsize=11, fontweight='bold')
    
    # Boundary protection
    gate_box = draw_box(ax, 8.5, 2, 3.5, 5,
                       'Boundary\nProtection\nGate',
                       COLORS['enhanced'], fontsize=11, fontweight='bold')
    
    # Output
    draw_box(ax, 13, 2, 0.8, 5,
            'Enhanced\nFeatures',
            COLORS['output'], fontsize=9, fontweight='bold')
    
    # Arrows
    for y in y_positions:
        draw_arrow(ax, (3, y+0.6), (4, y+0.6))
        draw_arrow(ax, (7.5, y+0.6), (8.5, y+0.6))
        draw_arrow(ax, (12, y+0.6), (13, y+0.6))
    
    # Annotation
    ax.text(5.75, 1, 'Each scale attends to other scales', 
            ha='center', va='center', fontsize=9, style='italic')
    ax.text(10.25, 1, 'Protect boundary information', 
            ha='center', va='center', fontsize=9, style='italic')
    
    output_path = 'draw/img/multiscale_fusion.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ MultiScale Fusion saved to {output_path}")
    plt.close()


def draw_c1c2_fusion():
    """Draw Enhanced C1C2 Fusion module"""
    fig, ax = setup_figure(14, 9)
    
    ax.text(7, 8.3, 'Enhanced C1C2 Fusion Module', 
            ha='center', va='center',
            fontsize=18, fontweight='bold')
    
    # Input
    c1_box = draw_box(ax, 1, 5, 2.5, 1.5, 'C1\n1/4 scale',
                     COLORS['input'], fontsize=10, fontweight='bold')
    c2_box = draw_box(ax, 1, 2.5, 2.5, 1.5, 'C2\n1/8 scale',
                     COLORS['input'], fontsize=10, fontweight='bold')
    
    # Upsample
    upsample_box = draw_box(ax, 4.5, 2.5, 2, 1.5, 'Upsample\n(Bilinear)',
                           COLORS['adapter'], fontsize=9)
    
    # Attention modules
    ch_attn = draw_box(ax, 7.5, 5, 2.5, 1.5, 'Channel\nAttention',
                      COLORS['fusion'], fontsize=10, fontweight='bold')
    sp_attn = draw_box(ax, 7.5, 2.5, 2.5, 1.5, 'Spatial\nAttention',
                      COLORS['fusion'], fontsize=10, fontweight='bold')
    
    # Gate
    gate_box = draw_box(ax, 11, 3.5, 2, 2, 'Gated\nFusion',
                       COLORS['enhanced'], fontsize=11, fontweight='bold')
    
    # Output
    output_box = draw_box(ax, 14, 4, 1.5, 1.5, 'Fused\nC1',
                        COLORS['output'], fontsize=10, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, (3.5, 5.75), (7.5, 5.75))
    draw_arrow(ax, (3.5, 3.25), (4.5, 3.25))
    draw_arrow(ax, (6.5, 3.25), (7.5, 3.25))
    draw_arrow(ax, (10, 5.75), (11, 5))
    draw_arrow(ax, (10, 3.25), (11, 4.5))
    draw_arrow(ax, (13, 4.75), (14, 4.75))
    
    # Residual connection
    draw_arrow(ax, (3.5, 5.75), (14, 4.75),
              color=COLORS['highlight'], style='dashed', linewidth=1.5)
    ax.text(8.75, 6.5, 'Residual', ha='center', va='center',
           fontsize=8, style='italic', color=COLORS['highlight'])
    
    ax.text(7, 1, 'Replaces simple upsampling + addition', 
            ha='center', va='center', fontsize=9, style='italic')
    
    output_path = 'draw/img/c1c2_fusion.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ C1C2 Fusion saved to {output_path}")
    plt.close()


def draw_text_attention():
    """Draw Text Attention Aggregation module"""
    fig, ax = setup_figure(14, 9)
    
    ax.text(7, 8.3, 'Text Attention Aggregation Module', 
            ha='center', va='center',
            fontsize=18, fontweight='bold')
    
    # Input
    text_input = draw_box(ax, 1, 4.5, 2.5, 2.5, 'Text Features\n[B×N×D]',
                         COLORS['text'], fontsize=10, fontweight='bold')
    
    # Base token
    base_box = draw_box(ax, 4.5, 6, 2.5, 1.2, 'Base Token\n(EOS/CLS)',
                      COLORS['input'], fontsize=9)
    
    # Attention
    attn_box = draw_box(ax, 4.5, 4, 2.5, 1.5, 'Multi-Head\nAttention',
                       COLORS['fusion'], fontsize=10, fontweight='bold')
    
    # Learnable query
    query_box = draw_box(ax, 4.5, 1.5, 2.5, 1.5, 'Learnable\nQuery',
                        COLORS['enhanced'], fontsize=9)
    
    # Fusion
    fusion_box = draw_box(ax, 8, 3.5, 3, 2.5, 'Weighted\nFusion\n(Learnable)',
                        COLORS['enhanced'], fontsize=10, fontweight='bold')
    
    # Output
    output_box = draw_box(ax, 12, 4.25, 1.5, 1.5, 'Aggregated\nText Feature',
                         COLORS['output'], fontsize=9, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, (3.5, 6.6), (4.5, 6.6))
    draw_arrow(ax, (3.5, 4.75), (4.5, 4.75))
    draw_arrow(ax, (3.5, 2.25), (4.5, 2.25))
    draw_arrow(ax, (7, 6.6), (8, 5.5))
    draw_arrow(ax, (7, 4.75), (8, 4.75))
    draw_arrow(ax, (11, 4.75), (12, 4.75))
    
    ax.text(7, 1, 'Replaces simple token selection (EOS/CLS)', 
            ha='center', va='center', fontsize=9, style='italic')
    
    output_path = 'draw/img/text_attention.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Text Attention saved to {output_path}")
    plt.close()


def draw_enhanced_loss():
    """Draw Enhanced Loss Functions module"""
    fig, ax = setup_figure(16, 9)
    
    ax.text(8, 8.3, 'Enhanced Loss Functions Module', 
            ha='center', va='center',
            fontsize=18, fontweight='bold')
    
    # Input
    pred_box = draw_box(ax, 1, 5.5, 2.5, 1.5, 'Predicted\nMask',
                      COLORS['input'], fontsize=10, fontweight='bold')
    target_box = draw_box(ax, 1, 3, 2.5, 1.5, 'Ground\nTruth',
                         COLORS['input'], fontsize=10, fontweight='bold')
    
    # Loss functions
    ce_box = draw_box(ax, 4.5, 6, 2, 1, 'CE Loss',
                     COLORS['loss'], fontsize=9)
    dice_box = draw_box(ax, 4.5, 4.5, 2, 1, 'Dice Loss',
                       COLORS['loss'], fontsize=9)
    focal_box = draw_box(ax, 4.5, 3, 2, 1, 'Focal Loss',
                        COLORS['enhanced'],
                        edgecolor=COLORS['highlight'],
                        fontsize=9, fontweight='bold')
    
    iou_box = draw_box(ax, 7.5, 6, 2, 1, 'IoU Loss',
                      COLORS['enhanced'],
                      edgecolor=COLORS['highlight'],
                      fontsize=9, fontweight='bold')
    boundary_box = draw_box(ax, 7.5, 4.5, 2, 1, 'Boundary Loss',
                           COLORS['enhanced'],
                           edgecolor=COLORS['highlight'],
                           fontsize=9, fontweight='bold')
    
    # Adaptive weighting
    adaptive_box = draw_box(ax, 10.5, 3.5, 3.5, 3, 'Adaptive\nWeighting\n(Learnable)',
                           COLORS['fusion'], fontsize=11, fontweight='bold')
    
    # Curriculum learning
    curriculum_box = draw_box(ax, 10.5, 0.5, 3.5, 2.5, 'Curriculum\nLearning',
                             COLORS['adapter'], fontsize=10, fontweight='bold')
    
    # Total loss
    total_box = draw_box(ax, 15, 3.5, 1, 3, 'Total\nLoss',
                        COLORS['output'], fontsize=11, fontweight='bold')
    
    # Arrows
    draw_arrow(ax, (3.5, 6.25), (4.5, 6.5))
    draw_arrow(ax, (3.5, 3.75), (4.5, 5))
    draw_arrow(ax, (3.5, 3.75), (4.5, 4))
    draw_arrow(ax, (6.5, 6.5), (7.5, 6.5))
    draw_arrow(ax, (6.5, 5), (7.5, 5))
    draw_arrow(ax, (6.5, 4), (7.5, 4))
    draw_arrow(ax, (9.5, 5), (10.5, 5))
    draw_arrow(ax, (14, 5), (15, 5))
    
    output_path = 'draw/img/enhanced_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Enhanced Loss saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Drawing Professional Architecture Diagrams")
    print("Style: Top-tier CV Conference (CVPR, ICCV, NeurIPS)")
    print("=" * 60)
    
    draw_overall_architecture()
    draw_multiscale_fusion()
    draw_c1c2_fusion()
    draw_text_attention()
    draw_enhanced_loss()
    
    print("\n" + "=" * 60)
    print("All diagrams saved to draw/img/")
    print("=" * 60)

