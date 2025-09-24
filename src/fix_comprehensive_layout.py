#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Comprehensive Performance Analysis Layout
重新设计布局，解决子图重叠问题
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置高质量绘图参数
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (20, 12),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 0.5,
})

# 模型颜色映射
MODEL_COLORS = {
    'GPT-Image-1': '#1f77b4',     
    'Sora': '#ff7f0e',            
    'Midjourney': '#2ca02c',      
    'DALL-E-3': '#d62728',        
    'SD15': '#9467bd'             
}

def create_gradient_background(ax, colors, direction='vertical'):
    """创建渐变背景"""
    gradient = LinearSegmentedColormap.from_list('gradient', colors)
    if direction == 'vertical':
        gradient_array = np.linspace(0, 1, 256).reshape(256, 1)
    else:
        gradient_array = np.linspace(0, 1, 256).reshape(1, 256)
    
    ax.imshow(gradient_array, aspect='auto', cmap=gradient, 
              extent=[ax.get_xlim()[0], ax.get_xlim()[1], 
                     ax.get_ylim()[0], ax.get_ylim()[1]], 
              alpha=0.1, zorder=0)

def create_fixed_comprehensive_performance_analysis():
    """创建布局优化的综合性能分析图"""
    
    # 数据准备
    models = ['GPT-Image-1', 'Sora', 'Midjourney', 'DALL-E-3', 'SD15']
    
    # 实际实验数据
    cot_scores = [7.557, 6.633, 6.710, 5.291, 5.000]
    niqe_scores = [4.513, 4.526, 2.645, 2.915, 3.408]
    brisque_scores = [14.224, 12.851, 19.419, 21.828, 20.135]
    is_scores = [14.155, 15.065, 10.613, 8.534, 10.292]
    piqe_scores = [33.179, 31.232, 42.842, 45.741, 43.449]
    perspective_scores = [0.860, 0.879, 0.900, 0.918, 0.862]
    circulation_scores = [0.349, 0.345, 0.297, 0.341, 0.345]
    
    # 创建更合理的布局 - 2x2网格，留足空间
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1],
                         hspace=0.35, wspace=0.3,
                         left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    fig.patch.set_facecolor('#fafafa')
    
    # === 子图1: 雷达图 (左上) ===
    ax_radar = fig.add_subplot(gs[0, 0], projection='polar')
    
    # 雷达图指标
    metrics = ['COT\nReasoning', 'NIQE\n(↓)', 'BRISQUE\n(↓)', 'IS\n(↑)', 'PIQE\n(↓)', 'Perspective\n(↑)', 'Circulation\n(↑)']
    
    # 数据标准化用于雷达图
    data_matrix = np.array([
        cot_scores,
        [1-x/25 for x in niqe_scores],  # NIQE反向标准化
        [1-x/100 for x in brisque_scores],  # BRISQUE反向标准化  
        [x/20 for x in is_scores],  # IS标准化
        [1-x/100 for x in piqe_scores],  # PIQE反向标准化
        perspective_scores,
        circulation_scores
    ]).T
    
    # 标准化到0-1
    data_matrix = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0))
    
    # 角度设置
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    # 绘制每个模型
    for i, model in enumerate(models):
        values = data_matrix[i].tolist()
        values += values[:1]  # 闭合
        
        # 绘制填充区域
        ax_radar.fill(angles, values, color=MODEL_COLORS[model], alpha=0.15, linewidth=0)
        # 绘制边界线
        ax_radar.plot(angles, values, color=MODEL_COLORS[model], linewidth=2.5, 
                     marker='o', markersize=6, markerfacecolor='white', 
                     markeredgecolor=MODEL_COLORS[model], markeredgewidth=2, label=model)
    
    # 设置标签
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics, fontsize=10, weight='bold')
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, alpha=0.7)
    ax_radar.grid(True, alpha=0.4, linewidth=1.2)
    ax_radar.set_facecolor('#ffffff')
    
    # 添加标题
    ax_radar.set_title('Model Performance Radar Chart', 
                      fontsize=16, weight='bold', pad=25,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='#2c3e50'))
    
    # 图例放在雷达图右侧
    ax_radar.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=True, 
                   fancybox=True, shadow=True, ncol=1, fontsize=10)
    
    # === 子图2: COT性能柱状图 (右上) ===
    ax_cot = fig.add_subplot(gs[0, 1])
    
    # 创建柱状图
    bars = ax_cot.bar(range(len(models)), cot_scores, color=[MODEL_COLORS[m] for m in models], 
                     alpha=0.8, edgecolor='white', linewidth=2, width=0.6)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, cot_scores)):
        height = bar.get_height()
        ax_cot.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                   f'{score:.3f}', ha='center', va='bottom', 
                   fontsize=10, weight='bold', color='#2c3e50')
    
    ax_cot.set_title('Chain-of-Thought Reasoning Scores', 
                    fontsize=14, weight='bold', pad=15)
    ax_cot.set_ylabel('COT Score', fontsize=12, weight='bold')
    ax_cot.set_xlabel('Models', fontsize=12, weight='bold')
    ax_cot.set_ylim(0, max(cot_scores) * 1.25)
    ax_cot.set_xticks(range(len(models)))
    ax_cot.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax_cot.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 背景渐变
    create_gradient_background(ax_cot, ['#ffffff', '#f8f9fa'])
    
    # === 子图3: 透视vs动线散点图 (左下) ===
    ax_scatter = fig.add_subplot(gs[1, 0])
    
    # 绘制散点
    for i, model in enumerate(models):
        ax_scatter.scatter(perspective_scores[i], circulation_scores[i], 
                          color=MODEL_COLORS[model], s=200, alpha=0.8, 
                          edgecolors='white', linewidths=2, zorder=5, label=model)
        
        # 添加模型标签 - 调整位置避免重叠
        offset_x = 0.005 if i % 2 == 0 else -0.005
        offset_y = 0.008 if i < 3 else -0.008
        
        ax_scatter.annotate(model, 
                           (perspective_scores[i] + offset_x, circulation_scores[i] + offset_y),
                           fontsize=9, weight='bold', ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor=MODEL_COLORS[model], alpha=0.8, 
                                   edgecolor='white'),
                           color='white')
    
    ax_scatter.set_xlabel('Perspective Consistency Score', fontsize=12, weight='bold')
    ax_scatter.set_ylabel('Circulation Rationality Score', fontsize=12, weight='bold') 
    ax_scatter.set_title('Geometric Precision vs Functional Rationality', 
                        fontsize=14, weight='bold', pad=15)
    ax_scatter.grid(True, alpha=0.3, linestyle='--')
    
    # 设置坐标轴范围，留出边距
    x_margin = (max(perspective_scores) - min(perspective_scores)) * 0.1
    y_margin = (max(circulation_scores) - min(circulation_scores)) * 0.1
    ax_scatter.set_xlim(min(perspective_scores) - x_margin, max(perspective_scores) + x_margin)
    ax_scatter.set_ylim(min(circulation_scores) - y_margin, max(circulation_scores) + y_margin)
    
    # 背景渐变
    create_gradient_background(ax_scatter, ['#ffffff', '#f8f9fa'])
    
    # === 子图4: 图像质量综合对比 (右下) ===
    ax_quality = fig.add_subplot(gs[1, 1])
    
    # 标准化质量指标 (归一化到0-1，越高越好)
    niqe_norm = [(25-x)/25 for x in niqe_scores]  # 反向标准化
    brisque_norm = [(100-x)/100 for x in brisque_scores]  # 反向标准化
    is_norm = [x/20 for x in is_scores]  # 标准化
    piqe_norm = [(100-x)/100 for x in piqe_scores]  # 反向标准化
    
    x = np.arange(len(models))
    width = 0.18  # 减小宽度避免重叠
    
    # 绘制分组柱状图 - 调整位置
    bars1 = ax_quality.bar(x - 1.5*width, niqe_norm, width, label='NIQE (↓)', 
                          color='#3498db', alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax_quality.bar(x - 0.5*width, brisque_norm, width, label='BRISQUE (↓)', 
                          color='#e74c3c', alpha=0.8, edgecolor='white', linewidth=1)
    bars3 = ax_quality.bar(x + 0.5*width, is_norm, width, label='IS (↑)', 
                          color='#2ecc71', alpha=0.8, edgecolor='white', linewidth=1)
    bars4 = ax_quality.bar(x + 1.5*width, piqe_norm, width, label='PIQE (↓)', 
                          color='#f39c12', alpha=0.8, edgecolor='white', linewidth=1)
    
    ax_quality.set_xlabel('Models', fontsize=12, weight='bold')
    ax_quality.set_ylabel('Normalized Score (Higher is Better)', fontsize=12, weight='bold')
    ax_quality.set_title('Image Quality Metrics Comparison', 
                        fontsize=14, weight='bold', pad=15)
    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax_quality.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=9)
    ax_quality.grid(axis='y', alpha=0.3, linestyle='--')
    ax_quality.set_ylim(0, 1.0)
    
    # 背景渐变
    create_gradient_background(ax_quality, ['#ffffff', '#f8f9fa'])
    
    # === 添加整体标题 ===
    fig.suptitle('Comprehensive Model Performance Analysis', 
                fontsize=20, weight='bold', y=0.96,
                bbox=dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.9, 
                         edgecolor='#2c3e50', linewidth=2))
    
    # 添加说明文字
    explanation = "Four-dimensional analysis: (1) Radar chart shows normalized performance across 7 metrics\n(2) Bar chart displays COT reasoning capabilities (3) Scatter plot reveals geometric vs functional trade-offs\n(4) Grouped bars compare image quality metrics (arrows indicate optimization direction)"
    
    fig.text(0.5, 0.02, explanation, ha='center', va='bottom', fontsize=10, 
            style='italic', alpha=0.8, wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12)  # 为标题和说明留出空间
    
    plt.savefig('figures/comprehensive_performance_analysis.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('figures/comprehensive_performance_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    """主函数：生成布局优化的综合性能分析图"""
    
    print("🔧 修复综合性能分析图布局...")
    
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    create_fixed_comprehensive_performance_analysis()
    
    print("✨ 布局优化完成！")
    print("📁 输出文件：")
    print("   - figures/comprehensive_performance_analysis.pdf")
    print("   - figures/comprehensive_performance_analysis.png")
    print("\n🎯 修复内容：")
    print("   - 调整子图间距，消除重叠")
    print("   - 优化图例和标签位置")
    print("   - 改善整体布局平衡")
    print("   - 增加说明文字")

if __name__ == "__main__":
    main()
