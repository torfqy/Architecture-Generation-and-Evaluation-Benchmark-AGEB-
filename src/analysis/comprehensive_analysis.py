#!/usr/bin/env python3
"""
Comprehensive Analysis and Visualization Script
Architecture Image Generation Model Comparison

This script analyzes and visualizes results from multiple image generation models:
- DALL-E-3
- GPT-Image-1  
- Midjourney (mj_imagine)
- Stable Diffusion 1.5 (SD15)
- Sora Image

Creates publication-quality visualizations and statistical summaries.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# Enhanced color palette with better contrast and accessibility
COLORS = {
    'DALL-E-3': '#1f77b4',      # Blue
    'GPT-Image-1': '#ff7f0e',   # Orange  
    'Midjourney': '#2ca02c',    # Green
    'SD15': '#d62728',          # Red
    'Sora': '#9467bd'           # Purple
}

# Circulation analysis colors
CIRCULATION_COLORS = {
    'efficiency': '#3498db',     # Blue
    'convenience': '#e74c3c',    # Red
    'dynamics': '#2ecc71',       # Green
    'overall': '#9b59b6'         # Purple
}

# Secondary colors for accents
ACCENT_COLORS = {
    'DALL-E-3': '#aec7e8',
    'GPT-Image-1': '#ffbb78',
    'Midjourney': '#98df8a',
    'SD15': '#ff9896',
    'Sora': '#c5b0d5'
}

def load_experiment_data():
    """Load data from all experiment directories."""
    base_dir = Path('/data_hdd/lx20/fqy_workspace/architecture_benchmark')
    
    experiments = {
        'DALL-E-3': base_dir / 'DALL-E-3',
        'GPT-Image-1': base_dir / 'gpt-image-1-results', 
        'Midjourney': base_dir / 'mj_imagine-results',
        'SD15': base_dir / 'SD15-results',
        'Sora': base_dir / 'sora_image-results'
    }
    
    data = {}
    summary_stats = {}
    
    for model_name, exp_dir in experiments.items():
        # Load summary statistics
        summary_file = exp_dir / 'summary.csv'
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
            summary_stats[model_name] = summary_df.iloc[0].to_dict()
        
        # Load detailed data
        final_json = exp_dir / 'final.json'
        if final_json.exists():
            with open(final_json, 'r', encoding='utf-8') as f:
                detailed_data = json.load(f)
            data[model_name] = detailed_data
            
    return data, summary_stats

def create_summary_table(summary_stats):
    """Create comprehensive summary statistics table."""
    df = pd.DataFrame(summary_stats).T
    
    # Calculate sample sizes
    sample_sizes = {
        'DALL-E-3': 300,  # Based on file analysis
        'GPT-Image-1': 300,
        'Midjourney': 138,  # Based on file size difference
        'SD15': 300,
        'Sora': 300
    }
    
    df['Sample_Size'] = df.index.map(sample_sizes)
    
    # Reorder columns for better presentation
    column_order = ['Sample_Size', 'cot_score', 'niqe', 'brisque', 'inception_score', 'piqe']
    df = df[column_order]
    
    # Round to 3 decimal places
    numeric_cols = ['cot_score', 'niqe', 'brisque', 'inception_score', 'piqe']
    df[numeric_cols] = df[numeric_cols].round(3)
    
    return df

def get_category_mapping():
    """Define category mapping for analysis."""
    return {
        1: "Object Counting",
        51: "Spatial Relations", 
        101: "Attribute Binding",
        151: "Complex Compositions",
        201: "Fine-grained Actions & Dynamic Layouts",
        251: "Negation Handling"
    }

def analyze_by_category(data):
    """Analyze performance by task category."""
    category_results = {}
    
    for model_name, model_data in data.items():
        if not model_data:
            continue
            
        categories = {}
        for item in model_data:
            category = item.get('category', 'Unknown')
            if category not in categories:
                categories[category] = {
                        'cot_scores': [],
                        'niqe': [],
                        'brisque': [], 
                        'inception_score': [],
                        'piqe': [],
                        'circulation_efficiency': [],
                        'circulation_convenience': [],
                        'circulation_dynamics': [],
                        'overall_circulation_score': []
                    }
            
            # Extract metrics
            cot_score = item.get('cot_score')
            if isinstance(cot_score, (int, float)):
                categories[category]['cot_scores'].append(cot_score)
                
            obj_metrics = item.get('objective_metrics', {})
            if obj_metrics:
                for metric in ['niqe', 'brisque', 'inception_score', 'piqe',
                              'circulation_efficiency', 'circulation_convenience',
                              'circulation_dynamics', 'overall_circulation_score']:
                    value = obj_metrics.get(metric)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        categories[category][metric].append(value)
        
        # Calculate category averages
        category_avg = {}
        for category, metrics in categories.items():
            category_avg[category] = {}
            for metric, values in metrics.items():
                if values:
                    category_avg[category][metric] = np.mean(values)
                else:
                    category_avg[category][metric] = np.nan
                    
        category_results[model_name] = category_avg
        
    return category_results

def create_visualizations(summary_df, category_results, data):
    """Create comprehensive visualization suite with enhanced scientific presentation."""
    
    # Create main comprehensive figure with better proportions
    fig = plt.figure(figsize=(20, 24), facecolor='white')
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1],
                         hspace=0.35, wspace=0.25)
    
    # Row 1: Core Performance Metrics
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    create_radar_chart(ax1, summary_df)
    
    ax2 = fig.add_subplot(gs[0, 1])
    create_cot_comparison(ax2, summary_df)
    
    ax3 = fig.add_subplot(gs[0, 2])
    create_quality_metrics_comparison(ax3, summary_df)
    
    # Row 2: Detailed Analysis
    ax4 = fig.add_subplot(gs[1, :2])
    create_category_heatmap(ax4, category_results)
    
    ax5 = fig.add_subplot(gs[1, 2])
    create_performance_vs_sample_size(ax5, summary_df)
    
    # Row 3: Statistical Analysis
    ax6 = fig.add_subplot(gs[2, :])
    create_enhanced_distributions(ax6, data)
    
    # Row 4: Summary and Trade-offs
    ax7 = fig.add_subplot(gs[3, :2])
    create_ranking_analysis(ax7, summary_df)
    
    ax8 = fig.add_subplot(gs[3, 2])
    create_tradeoff_analysis(ax8, summary_df)
    
    # Add main title with better positioning
    fig.suptitle('Architecture Image Generation Model Analysis\nComprehensive Performance Comparison', 
                 fontsize=18, fontweight='bold', y=0.97, color='#2c3e50')
    
    # Add subtle background
    fig.patch.set_facecolor('#fafafa')
    
    plt.savefig('comprehensive_model_analysis_enhanced.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('comprehensive_model_analysis_enhanced.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Create separate detailed analysis figures
    create_detailed_category_analysis(category_results, data)
    create_quality_metrics_deep_dive(data)
    create_circulation_analysis_visualization(data)  # New circulation analysis
    
    return fig

def create_radar_chart(ax, summary_df):
    """Create radar chart for overall performance comparison."""
    
    # Normalize metrics for radar chart (0-1 scale)
    metrics = ['cot_score', 'inception_score', 'niqe', 'brisque', 'piqe']
    metric_labels = ['COT Score', 'Inception\nScore', 'NIQE\n(inv)', 'BRISQUE\n(inv)', 'PIQE\n(inv)']
    
    # Invert metrics where lower is better
    normalized_data = summary_df.copy()
    normalized_data['cot_score'] = normalized_data['cot_score'] / 10.0  # Scale to 0-1
    normalized_data['inception_score'] = np.clip(normalized_data['inception_score'] / 20.0, 0, 1)
    normalized_data['niqe'] = 1 - np.clip(normalized_data['niqe'] / 10.0, 0, 1)  # Invert
    normalized_data['brisque'] = 1 - np.clip(normalized_data['brisque'] / 50.0, 0, 1)  # Invert
    normalized_data['piqe'] = 1 - np.clip(normalized_data['piqe'] / 100.0, 0, 1)  # Invert
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    for model in normalized_data.index:
        values = [normalized_data.loc[model, metric] for metric in metrics]
        values += values[:1]  # Close the plot
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, 
                color=COLORS.get(model, '#333333'), markersize=6)
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(model, '#333333'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance\nComparison', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, frameon=True, 
              fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_facecolor('#fafafa')

def create_cot_comparison(ax, summary_df):
    """Create COT score comparison bar chart."""
    models = summary_df.index
    scores = summary_df['cot_score']
    
    # Create bars with gradient effect
    bars = ax.bar(models, scores, 
                  color=[COLORS.get(m, '#333333') for m in models],
                  alpha=0.85, edgecolor='white', linewidth=2)
    
    # Add value labels on bars with better positioning
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, color='#2c3e50')
    
    ax.set_ylabel('Chain-of-Thought Score', fontweight='bold', fontsize=11)
    ax.set_title('COT Performance Comparison', fontweight='bold', fontsize=12, pad=15)
    ax.set_ylim(0, max(scores) * 1.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_facecolor('#fafafa')

def create_quality_metrics_comparison(ax, summary_df):
    """Create grouped bar chart for image quality metrics."""
    metrics = ['niqe', 'brisque', 'piqe']
    metric_colors = ['#e74c3c', '#f39c12', '#8e44ad']
    x = np.arange(len(summary_df.index))
    width = 0.25
    
    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, summary_df[metric], width, 
                     label=metric.upper(), alpha=0.8, color=color,
                     edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=11)
    ax.set_ylabel('Quality Score', fontweight='bold', fontsize=11)
    ax.set_title('Image Quality Metrics\n(Lower is Better)', fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.index, rotation=45, ha='right', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_facecolor('#fafafa')

def create_category_heatmap(ax, category_results):
    """Create heatmap showing performance across categories."""
    
    # Prepare data for heatmap with shorter category names
    categories = ['Object\nCounting', 'Spatial\nRelations', 'Attribute\nBinding', 
                 'Complex\nCompositions', 'Fine-grained\nActions', 'Negation\nHandling']
    full_categories = ['Object Counting', 'Spatial Relations', 'Attribute Binding', 
                      'Complex Compositions', 'Fine-grained Actions & Dynamic Layouts', 'Negation Handling']
    
    heatmap_data = []
    models = []
    
    for model, cat_data in category_results.items():
        if not cat_data:
            continue
        models.append(model)
        row = []
        for category in full_categories:
            cot_score = cat_data.get(category, {}).get('cot_scores', np.nan)
            if isinstance(cot_score, (list, np.ndarray)) and len(cot_score) > 0:
                row.append(np.mean(cot_score))
            elif isinstance(cot_score, (int, float)):
                row.append(cot_score)
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data, index=models, columns=categories)
        
        # Create heatmap with better styling
        sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', center=5, 
                   cbar_kws={'label': 'COT Score', 'shrink': 0.8}, 
                   ax=ax, fmt='.2f', linewidths=0.5, linecolor='white',
                   annot_kws={'fontsize': 9, 'fontweight': 'bold'})
        
        ax.set_title('Performance by Task Category', fontweight='bold', fontsize=12, pad=15)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontweight='bold', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontweight='bold', fontsize=10)

def create_metric_distributions(ax, summary_df):
    """Create violin plot showing metric distributions."""
    
    # Prepare data for violin plot
    plot_data = []
    for model in summary_df.index:
        for metric in ['cot_score', 'inception_score']:
            plot_data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Value': summary_df.loc[model, metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    sns.boxplot(data=plot_df, x='Metric', y='Value', hue='Model', ax=ax,
                palette=[COLORS.get(m, '#333333') for m in summary_df.index])
    
    ax.set_title('Performance Metric Distributions', fontweight='bold')
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

def create_correlation_matrix(ax, summary_df):
    """Create correlation matrix for all metrics."""
    
    metrics = ['cot_score', 'niqe', 'brisque', 'inception_score', 'piqe']
    corr_data = summary_df[metrics]
    
    correlation_matrix = corr_data.T.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Inter-Model Correlation Matrix', fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

def create_performance_vs_sample_size(ax, summary_df):
    """Analyze performance vs sample size relationship."""
    
    # Create scatter plot with better styling
    scatter = ax.scatter(summary_df['Sample_Size'], summary_df['cot_score'], 
                        s=300, alpha=0.8, 
                        c=[COLORS.get(m, '#333333') for m in summary_df.index],
                        edgecolors='white', linewidths=2)
    
    # Add model labels with better positioning
    for idx, model in enumerate(summary_df.index):
        ax.annotate(model, (summary_df.loc[model, 'Sample_Size'], summary_df.loc[model, 'cot_score']),
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=9, fontweight='bold', color='#2c3e50',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel('Sample Size', fontweight='bold', fontsize=11)
    ax.set_ylabel('COT Score', fontweight='bold', fontsize=11)
    ax.set_title('Performance vs\nSample Size', fontweight='bold', fontsize=12, pad=15)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#fafafa')
    
    # Add trend line if there's a correlation
    if len(summary_df) > 2:
        z = np.polyfit(summary_df['Sample_Size'], summary_df['cot_score'], 1)
        p = np.poly1d(z)
        ax.plot(summary_df['Sample_Size'], p(summary_df['Sample_Size']), 
                "r--", alpha=0.6, linewidth=2, label='Trend')

def create_statistical_analysis(ax, summary_df, data):
    """Create statistical significance analysis."""
    
    # Calculate confidence intervals and effect sizes
    stats_data = []
    for model, model_data in data.items():
        if model_data:
            cot_scores = [item.get('cot_score') for item in model_data 
                         if isinstance(item.get('cot_score'), (int, float))]
            if cot_scores:
                mean_score = np.mean(cot_scores)
                std_score = np.std(cot_scores)
                se_score = std_score / np.sqrt(len(cot_scores))
                ci_lower = mean_score - 1.96 * se_score
                ci_upper = mean_score + 1.96 * se_score
                
                stats_data.append({
                    'Model': model,
                    'Mean': mean_score,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'N': len(cot_scores)
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Create error bar plot
        ax.errorbar(range(len(stats_df)), stats_df['Mean'], 
                   yerr=[stats_df['Mean'] - stats_df['CI_Lower'], 
                         stats_df['CI_Upper'] - stats_df['Mean']],
                   fmt='o', capsize=5, capthick=2, markersize=8)
        
        ax.set_xticks(range(len(stats_df)))
        ax.set_xticklabels(stats_df['Model'], rotation=45, ha='right')
        ax.set_ylabel('COT Score')
        ax.set_title('Statistical Confidence Intervals\n(95% CI)', fontweight='bold')
        ax.grid(True, alpha=0.3)

def create_enhanced_distributions(ax, data):
    """Create enhanced distribution plots with violin plots."""
    
    plot_data = []
    model_order = []
    for model, model_data in data.items():
        if model_data and model not in model_order:
            model_order.append(model)
            for item in model_data:
                cot_score = item.get('cot_score')
                if isinstance(cot_score, (int, float)):
                    plot_data.append({'Model': model, 'COT_Score': cot_score})
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot with better styling
        box_plot = ax.boxplot([plot_df[plot_df['Model'] == model]['COT_Score'].values 
                              for model in model_order], 
                             positions=range(len(model_order)),
                             patch_artist=True, showmeans=True, 
                             meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 6},
                             medianprops={'color': 'black', 'linewidth': 2},
                             whiskerprops={'color': 'black', 'linewidth': 1.5},
                             capprops={'color': 'black', 'linewidth': 1.5})
        
        # Color the box plots
        for i, (patch, model) in enumerate(zip(box_plot['boxes'], model_order)):
            patch.set_facecolor(COLORS.get(model, '#333333'))
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Add sample size annotations
        for i, model in enumerate(model_order):
            n_samples = len(plot_df[plot_df['Model'] == model])
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n_samples}', 
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=45, ha='right', fontweight='bold')
        ax.set_ylabel('COT Score Distribution', fontweight='bold', fontsize=11)
        ax.set_title('Performance Score Distributions\n(Box Plot with Sample Sizes)', fontweight='bold', fontsize=12, pad=15)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#fafafa')

def create_ranking_analysis(ax, summary_df):
    """Create comprehensive ranking analysis."""
    
    # Calculate Quality_Score if not exists
    if 'Quality_Score' not in summary_df.columns:
        summary_df['Quality_Score'] = (
            (1 - summary_df['niqe'] / summary_df['niqe'].max()) * 0.25 +
            (1 - summary_df['brisque'] / summary_df['brisque'].max()) * 0.25 +
            (summary_df['inception_score'] / summary_df['inception_score'].max()) * 0.25 +
            (1 - summary_df['piqe'] / summary_df['piqe'].max()) * 0.25
        )
    
    metrics = ['cot_score', 'Quality_Score']
    metric_names = ['COT Performance', 'Image Quality']
    
    x = np.arange(len(summary_df.index))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c']  # Blue for COT, Red for Quality
    
    for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        # Normalize rankings (1 = best, higher number = worse)
        ranks = summary_df[metric].rank(ascending=False)
        offset = (i - 0.5) * width
        
        bars = ax.bar(x + offset, ranks, width, label=name, alpha=0.8,
                     color=color, edgecolor='white', linewidth=1)
        
        # Add rank labels on bars with better styling
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                   f'#{int(rank)}', ha='center', va='center', 
                   fontweight='bold', fontsize=9, color='white')
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=11)
    ax.set_ylabel('Ranking (1 = Best)', fontweight='bold', fontsize=11)
    ax.set_title('Model Rankings Comparison', fontweight='bold', fontsize=12, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df.index, rotation=45, ha='right', fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_facecolor('#fafafa')
    ax.invert_yaxis()  # Best rank (1) at top

def create_tradeoff_analysis(ax, summary_df):
    """Create trade-off analysis between quality and performance."""
    
    # Calculate Quality_Score if not exists
    if 'Quality_Score' not in summary_df.columns:
        summary_df['Quality_Score'] = (
            (1 - summary_df['niqe'] / summary_df['niqe'].max()) * 0.25 +
            (1 - summary_df['brisque'] / summary_df['brisque'].max()) * 0.25 +
            (summary_df['inception_score'] / summary_df['inception_score'].max()) * 0.25 +
            (1 - summary_df['piqe'] / summary_df['piqe'].max()) * 0.25
        )
    
    x = summary_df['Quality_Score']
    y = summary_df['cot_score']
    
    # Scale bubble sizes for better visibility
    sizes = (summary_df['Sample_Size'] / summary_df['Sample_Size'].max()) * 500 + 100
    
    scatter = ax.scatter(x, y, s=sizes, 
                        c=[COLORS.get(m, '#333333') for m in summary_df.index],
                        alpha=0.7, edgecolors='white', linewidths=2)
    
    # Add model labels with better styling
    for model in summary_df.index:
        ax.annotate(model, (summary_df.loc[model, 'Quality_Score'], 
                           summary_df.loc[model, 'cot_score']),
                   xytext=(8, 8), textcoords='offset points', 
                   fontsize=9, fontweight='bold', color='#2c3e50',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                            alpha=0.8, edgecolor='gray'))
    
    # Add trend line with correlation coefficient
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Calculate and display correlation
    correlation = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Image Quality Score', fontweight='bold', fontsize=11)
    ax.set_ylabel('COT Performance Score', fontweight='bold', fontsize=11)
    ax.set_title('Performance vs Quality\nTrade-off Analysis', fontweight='bold', fontsize=12, pad=15)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#fafafa')

def create_summary_table_viz(ax, summary_df):
    """Create a visual summary table."""
    
    ax.axis('off')
    
    # Prepare table data
    table_data = summary_df.round(3).values
    col_labels = ['Sample Size', 'COT Score', 'NIQE↓', 'BRISQUE↓', 'Inception↑', 'PIQE↓', 'COT Rank', 'Quality Score', 'Quality Rank']
    row_labels = summary_df.index
    
    # Create table
    table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code the cells based on performance
    for i, model in enumerate(row_labels):
        # Color header row
        table[(0, i)].set_facecolor(COLORS.get(model, '#DDDDDD'))
        table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color best values in each column
        for j in range(len(col_labels)):
            if j in [1, 4, 7]:  # Higher is better columns
                if summary_df.iloc[i, j] == summary_df.iloc[:, j].max():
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
            elif j in [2, 3, 5]:  # Lower is better columns  
                if summary_df.iloc[i, j] == summary_df.iloc[:, j].min():
                    table[(i+1, j)].set_facecolor('#90EE90')  # Light green
    
    ax.set_title('Comprehensive Performance Summary\n(Green = Best Performance)', 
                fontweight='bold', pad=20, fontsize=12)

def create_detailed_category_analysis(category_results, data):
    """Create detailed category-wise analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
    axes = axes.flatten()
    
    categories = ['Object Counting', 'Spatial Relations', 'Attribute Binding', 
                 'Complex Compositions', 'Fine-grained Actions & Dynamic Layouts', 'Negation Handling']
    
    # Color gradient for better visual distinction
    category_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
    
    for i, category in enumerate(categories):
        ax = axes[i]
        
        category_scores = {}
        for model, cat_data in category_results.items():
            score = cat_data.get(category, {}).get('cot_scores', np.nan)
            if not np.isnan(score):
                category_scores[model] = score
        
        if category_scores:
            models = list(category_scores.keys())
            scores = list(category_scores.values())
            
            # Use gradient colors for each category
            bars = ax.bar(models, scores, 
                         color=[COLORS.get(m, '#333333') for m in models], 
                         alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add value labels with better styling
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{score:.1f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10, color='#2c3e50')
            
            # Highlight the best performer
            best_idx = scores.index(max(scores))
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        # Improved styling
        ax.set_title(category, fontweight='bold', fontsize=11, pad=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=category_colors[i], alpha=0.2))
        ax.set_ylabel('COT Score', fontweight='bold', fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#fafafa')
        ax.set_ylim(0, 10)
        
        # Add performance indicators
        if category_scores:
            avg_score = np.mean(list(category_scores.values()))
            ax.axhline(y=avg_score, color='red', linestyle=':', alpha=0.7, linewidth=2)
            ax.text(0.02, 0.98, f'Avg: {avg_score:.1f}', transform=ax.transAxes, 
                   fontsize=9, fontweight='bold', va='top',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    plt.suptitle('Detailed Category Performance Analysis\nCOT Scores by Task Category (Gold border = Best)', 
                 fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('detailed_category_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('detailed_category_analysis.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_quality_metrics_deep_dive(data):
    """Create deep dive analysis of quality metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='white')
    
    metrics = ['niqe', 'brisque', 'inception_score', 'piqe']
    metric_titles = ['NIQE (Lower is Better)', 'BRISQUE (Lower is Better)', 
                    'Inception Score (Higher is Better)', 'PIQE (Lower is Better)']
    metric_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    for idx, (metric, title, color) in enumerate(zip(metrics, metric_titles, metric_colors)):
        ax = axes[idx // 2, idx % 2]
        
        plot_data = []
        for model, model_data in data.items():
            if model_data:
                for item in model_data:
                    obj_metrics = item.get('objective_metrics', {})
                    value = obj_metrics.get(metric)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        plot_data.append({'Model': model, 'Value': value})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            model_list = sorted(plot_df['Model'].unique())
            
            # Create violin plot for better distribution visualization
            parts = ax.violinplot([plot_df[plot_df['Model'] == model]['Value'].values 
                                  for model in model_list], 
                                 positions=range(len(model_list)), 
                                 showmeans=True, showmedians=True, showextrema=True)
            
            # Color the violin plots
            for i, pc in enumerate(parts['bodies']):
                model = model_list[i]
                pc.set_facecolor(COLORS.get(model, '#333333'))
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Style the statistical lines
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    parts[partname].set_edgecolor('black')
                    parts[partname].set_linewidth(2)
            
            # Add sample size annotations
            for i, model in enumerate(model_list):
                n_samples = len(plot_df[plot_df['Model'] == model])
                mean_val = plot_df[plot_df['Model'] == model]['Value'].mean()
                ax.text(i, ax.get_ylim()[1] * 0.9, f'n={n_samples}', 
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                ax.text(i, ax.get_ylim()[1] * 0.8, f'μ={mean_val:.1f}', 
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
            
            ax.set_xticks(range(len(model_list)))
            ax.set_xticklabels(model_list, rotation=45, ha='right', fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=12, pad=15,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
        ax.set_ylabel('Score Value', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#fafafa')
    
    plt.suptitle('Quality Metrics Deep Dive Analysis\nDistribution Patterns and Statistical Properties', 
                 fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('quality_metrics_deep_dive.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('quality_metrics_deep_dive.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def create_circulation_analysis_visualization(data):
    """Create comprehensive circulation analysis visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    
    circulation_metrics = ['circulation_efficiency', 'circulation_convenience', 
                          'circulation_dynamics', 'overall_circulation_score']
    metric_titles = ['Circulation Efficiency', 'Circulation Convenience',
                    'Circulation Dynamics', 'Overall Circulation Score']
    metric_colors = [CIRCULATION_COLORS['efficiency'], CIRCULATION_COLORS['convenience'],
                    CIRCULATION_COLORS['dynamics'], CIRCULATION_COLORS['overall']]
    
    for idx, (metric, title, color) in enumerate(zip(circulation_metrics, metric_titles, metric_colors)):
        ax = axes[idx // 2, idx % 2]
        
        plot_data = []
        for model, model_data in data.items():
            if model_data:
                for item in model_data:
                    obj_metrics = item.get('objective_metrics', {})
                    value = obj_metrics.get(metric)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        plot_data.append({'Model': model, 'Value': value})
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            model_list = sorted(plot_df['Model'].unique())
            
            # Create box plot for distribution visualization
            box_data = [plot_df[plot_df['Model'] == model]['Value'].values 
                       for model in model_list]
            
            box_plot = ax.boxplot(box_data, positions=range(len(model_list)),
                                 patch_artist=True, showmeans=True,
                                 meanprops={'marker': 'D', 'markerfacecolor': color, 
                                           'markeredgecolor': color, 'markersize': 6},
                                 medianprops={'color': 'black', 'linewidth': 2},
                                 whiskerprops={'color': 'black', 'linewidth': 1.5},
                                 capprops={'color': 'black', 'linewidth': 1.5})
            
            # Color the box plots
            for i, (patch, model) in enumerate(zip(box_plot['boxes'], model_list)):
                patch.set_facecolor(COLORS.get(model, color))
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)
            
            # Add statistics annotations
            for i, model in enumerate(model_list):
                model_data = plot_df[plot_df['Model'] == model]['Value']
                n_samples = len(model_data)
                mean_val = model_data.mean()
                std_val = model_data.std()
                
                ax.text(i, ax.get_ylim()[1] * 0.9, f'n={n_samples}', 
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                ax.text(i, ax.get_ylim()[1] * 0.8, f'μ={mean_val:.3f}', 
                       ha='center', va='top', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3))
                if not np.isnan(std_val):
                    ax.text(i, ax.get_ylim()[1] * 0.7, f'σ={std_val:.3f}', 
                           ha='center', va='top', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.5))
            
            ax.set_xticks(range(len(model_list)))
            ax.set_xticklabels(model_list, rotation=45, ha='right', fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=12, pad=15,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
        ax.set_ylabel('Score (0.0 - 1.0)', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#fafafa')
        ax.set_ylim(0, 1.0)  # Circulation scores are normalized to 0-1
    
    plt.suptitle('Circulation Analysis Performance Comparison\nArchitectural Design Flow Rationality Assessment', 
                 fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('circulation_analysis_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('circulation_analysis_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Create circulation correlation heatmap
    create_circulation_correlation_heatmap(data)

def create_circulation_correlation_heatmap(data):
    """Create correlation heatmap between circulation metrics and other performance indicators."""
    
    # Collect all metrics for correlation analysis
    all_metrics = []
    for model, model_data in data.items():
        if model_data:
            for item in model_data:
                metrics_dict = {'model': model}
                
                # COT score
                cot_score = item.get('cot_score')
                if isinstance(cot_score, (int, float)):
                    metrics_dict['cot_score'] = cot_score
                
                # Objective metrics
                obj_metrics = item.get('objective_metrics', {})
                for metric in ['niqe', 'brisque', 'inception_score', 'piqe',
                              'circulation_efficiency', 'circulation_convenience',
                              'circulation_dynamics', 'overall_circulation_score']:
                    value = obj_metrics.get(metric)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics_dict[metric] = value
                
                if len(metrics_dict) > 2:  # At least model + 1 metric
                    all_metrics.append(metrics_dict)
    
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Select numeric columns for correlation
        numeric_cols = ['cot_score', 'niqe', 'brisque', 'inception_score', 'piqe',
                       'circulation_efficiency', 'circulation_convenience',
                       'circulation_dynamics', 'overall_circulation_score']
        
        available_cols = [col for col in numeric_cols if col in metrics_df.columns]
        
        if len(available_cols) > 1:
            corr_data = metrics_df[available_cols].corr()
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            
            sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=ax, mask=mask,
                       cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
                       fmt='.3f', linewidths=0.5, linecolor='white',
                       annot_kws={'fontsize': 10, 'fontweight': 'bold'})
            
            ax.set_title('Circulation Metrics Correlation Analysis\nRelationships between Flow Rationality and Other Performance Indicators', 
                        fontweight='bold', fontsize=14, pad=20)
            
            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
            plt.setp(ax.get_yticklabels(), rotation=0, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('circulation_correlation_heatmap.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            plt.savefig('circulation_correlation_heatmap.pdf', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            plt.close()

def save_summary_table(summary_df):
    """Save summary table in multiple formats."""
    
    # Enhanced table with additional statistics
    enhanced_df = summary_df.copy()
    
    # Add ranking columns
    enhanced_df['COT_Rank'] = enhanced_df['cot_score'].rank(ascending=False).astype(int)
    enhanced_df['Quality_Score'] = (
        (1 - enhanced_df['niqe'] / enhanced_df['niqe'].max()) * 0.25 +
        (1 - enhanced_df['brisque'] / enhanced_df['brisque'].max()) * 0.25 +
        (enhanced_df['inception_score'] / enhanced_df['inception_score'].max()) * 0.25 +
        (1 - enhanced_df['piqe'] / enhanced_df['piqe'].max()) * 0.25
    )
    enhanced_df['Quality_Rank'] = enhanced_df['Quality_Score'].rank(ascending=False).astype(int)
    
    # Save to CSV
    enhanced_df.to_csv('model_comparison_summary.csv')
    
    # Save to LaTeX for scientific papers
    latex_table = enhanced_df.to_latex(
        float_format=lambda x: f'{x:.3f}' if isinstance(x, float) else str(x),
        caption='Comprehensive Model Performance Comparison',
        label='tab:model_comparison'
    )
    
    with open('model_comparison_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("Summary table saved:")
    print("- CSV: model_comparison_summary.csv") 
    print("- LaTeX: model_comparison_table.tex")
    
    return enhanced_df

def main():
    """Main analysis function."""
    print("Loading experiment data...")
    data, summary_stats = load_experiment_data()
    
    print("Creating summary table...")
    summary_df = create_summary_table(summary_stats)
    
    print("Analyzing by category...")
    category_results = analyze_by_category(data)
    
    print("Creating visualizations...")
    fig = create_visualizations(summary_df, category_results, data)
    
    print("Saving summary table...")
    enhanced_df = save_summary_table(summary_df)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL ANALYSIS RESULTS")
    print("="*60)
    print(enhanced_df)
    print("="*60)
    
    print(f"\nVisualization saved: comprehensive_model_analysis.png")
    print(f"High-quality PDF: comprehensive_model_analysis.pdf")
    
    # Calculate key insights
    best_cot = enhanced_df['cot_score'].idxmax()
    best_quality = enhanced_df['Quality_Score'].idxmax()
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"🏆 Best COT Performance: {best_cot} ({enhanced_df.loc[best_cot, 'cot_score']:.3f})")
    print(f"🎨 Best Image Quality: {best_quality} ({enhanced_df.loc[best_quality, 'Quality_Score']:.3f})")
    print(f"📈 Largest Sample: {enhanced_df['Sample_Size'].idxmax()} ({enhanced_df['Sample_Size'].max()} samples)")
    
    return enhanced_df, fig

if __name__ == "__main__":
    enhanced_df, fig = main()
