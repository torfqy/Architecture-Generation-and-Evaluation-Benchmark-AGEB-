#!/usr/bin/env python3
"""
Generate perspective analysis process images using real dataset results.
All text in English for international publication.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import math
import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Any
from perspective_analysis import PerspectiveAnalyzer

def load_real_data():
    """Load real evaluation results from all models."""
    all_results = {}
    models = ['DALL-E-3', 'gpt-image-1-results', 'mj_imagine-results', 'SD15-results', 'sora_image-results']
    
    for model in models:
        try:
            # Load final.json
            final_path = f"{model}/final.json"
            if os.path.exists(final_path):
                with open(final_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_results[model] = data
                print(f"✅ Loaded {len(data)} results from {model}")
            else:
                print(f"❌ No data found for {model}")
        except Exception as e:
            print(f"❌ Error loading {model}: {e}")
    
    return all_results

def find_best_perspective_examples(all_results, top_k=5):
    """Find examples with best perspective scores for visualization."""
    best_examples = []
    
    for model, results in all_results.items():
        for item in results:
            if 'objective_metrics' in item and 'perspective_score' in item['objective_metrics']:
                score = item['objective_metrics']['perspective_score']
                
                # Check if image exists
                image_path = f"{model}/images/{item['row_id']}.png"
                if not os.path.exists(image_path):
                    image_path = f"{model}/images/{item['row_id']}.webp"
                
                if os.path.exists(image_path):
                    best_examples.append({
                        'model': model,
                        'row_id': item['row_id'],
                        'image_path': image_path,
                        'perspective_score': score,
                        'prompt': item.get('prompt', ''),
                        'category': item.get('category', ''),
                        'metrics': item['objective_metrics']
                    })
    
    # Sort by perspective score and take top k
    best_examples.sort(key=lambda x: x['perspective_score'], reverse=True)
    return best_examples[:top_k]

def create_real_data_process_visualization(best_examples):
    """Create visualization using real data examples."""
    
    # Create figure directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Initialize the analyzer
    analyzer = PerspectiveAnalyzer()
    
    # Use the best example for detailed process visualization
    if not best_examples:
        print("❌ No valid examples found")
        return
    
    best_example = best_examples[0]
    image_path = best_example['image_path']
    
    print(f"🎯 Using best example: {best_example['model']} - Image {best_example['row_id']}")
    print(f"📊 Perspective Score: {best_example['perspective_score']:.4f}")
    print(f"📝 Prompt: {best_example['prompt'][:100]}...")
    
    # Load the image
    image = analyzer.load_image_from_path(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # Step 1: Original Image
    ax1 = plt.subplot(4, 2, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title('Step 1: Real Generated Architectural Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Add image info
    info_text = f'Model: {best_example["model"]}\n'
    info_text += f'Category: {best_example["category"]}\n'
    info_text += f'Resolution: {image.shape[1]}×{image.shape[0]}\n'
    info_text += f'Perspective Score: {best_example["perspective_score"]:.4f}'
    
    ax1.text(0.02, 0.98, info_text, 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Step 2: Edge Detection
    ax2 = plt.subplot(4, 2, 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Step 2: Canny Edge Detection', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add parameters annotation
    ax2.text(0.02, 0.98, 'Canny Parameters:\nLow threshold: 50\nHigh threshold: 150\nGaussian blur: 5×5', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Step 3: Line Detection
    ax3 = plt.subplot(4, 2, 3)
    lines = analyzer.detect_lines(image)
    line_image = image_rgb.copy()
    
    # Draw detected lines
    for i, (x1, y1, x2, y2) in enumerate(lines[:100]):  # Limit to first 100 lines for clarity
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    ax3.imshow(line_image)
    ax3.set_title('Step 3: Hough Line Detection', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add detection stats
    stats_text = f'Detection Parameters:\nThreshold: 100\nMin line length: 50px\nMax line gap: 10px\nDetected lines: {len(lines)}'
    ax3.text(0.02, 0.98, stats_text, 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Step 4: Vanishing Point Detection
    ax4 = plt.subplot(4, 2, 4)
    vanishing_points = analyzer.detect_vanishing_points(lines, image.shape)
    vp_image = image_rgb.copy()
    
    # Draw lines grouped by angle
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    angle_groups = {}
    angle_tolerance = np.pi / 18  # 10 degrees
    
    for line in lines:
        x1, y1, x2, y2 = line
        angle = analyzer.calculate_line_angle(x1, y1, x2, y2)
        if angle < 0:
            angle += np.pi
        
        # Find existing group or create new one
        assigned = False
        for group_angle in angle_groups.keys():
            if abs(angle - group_angle) < angle_tolerance:
                angle_groups[group_angle].append(line)
                assigned = True
                break
        
        if not assigned:
            angle_groups[angle] = [line]
    
    # Draw grouped lines with different colors
    color_idx = 0
    for group_angle, group_lines in angle_groups.items():
        if len(group_lines) >= 2:  # Only show groups with multiple lines
            color = colors[color_idx % len(colors)]
            for x1, y1, x2, y2 in group_lines:
                cv2.line(vp_image, (x1, y1), (x2, y2), color, 2)
            color_idx += 1
    
    # Draw vanishing points
    for vp_x, vp_y in vanishing_points:
        if 0 <= vp_x < image.shape[1] and 0 <= vp_y < image.shape[0]:
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 10, (255, 255, 255), -1)
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 10, (0, 0, 0), 3)
    
    ax4.imshow(vp_image)
    ax4.set_title('Step 4: Vanishing Point Detection', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Add vanishing point stats
    vp_stats = f'Angle tolerance: 10°\nParallel groups: {len([g for g in angle_groups.values() if len(g) >= 2])}\nVanishing points: {len(vanishing_points)}'
    ax4.text(0.02, 0.98, vp_stats, 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Step 5: Perspective Metrics Calculation
    ax5 = plt.subplot(4, 2, (5, 6))
    
    # Calculate all metrics
    vanishing_consistency = analyzer.calculate_vanishing_point_consistency(lines, vanishing_points)
    proportion_consistency = analyzer.calculate_proportion_consistency(lines)
    parallel_consistency = analyzer.calculate_parallel_line_consistency(lines)
    vertical_preservation = analyzer.calculate_vertical_line_preservation(lines)
    horizontal_preservation = analyzer.calculate_horizontal_line_preservation(lines)
    
    # Create bar chart of metrics
    metrics = {
        'Vanishing Point\nConsistency': vanishing_consistency,
        'Proportion\nConsistency': proportion_consistency,
        'Parallel Line\nConsistency': parallel_consistency,
        'Vertical Line\nPreservation': vertical_preservation,
        'Horizontal Line\nPreservation': horizontal_preservation
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(metric_names))
    bars = ax5.barh(y_pos, metric_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    
    # Add weight annotations
    for i, (bar, weight, value) in enumerate(zip(bars, weights, metric_values)):
        ax5.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}\n(Weight: {weight:.0%})', 
                va='center', ha='left', fontsize=10)
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(metric_names)
    ax5.set_xlabel('Score', fontsize=12)
    ax5.set_title('Step 5: Real Perspective Consistency Metrics', fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 1.1)
    ax5.grid(axis='x', alpha=0.3)
    
    # Calculate final score and compare with stored score
    calculated_score = sum(v * w for v, w in zip(metric_values, weights))
    stored_score = best_example['perspective_score']
    
    final_text = f'Calculated Score: {calculated_score:.3f}\nStored Score: {stored_score:.3f}\nDifference: {abs(calculated_score - stored_score):.3f}'
    ax5.text(0.5, -0.15, final_text, 
             transform=ax5.transAxes, fontsize=12, fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="orange", alpha=0.8))
    
    # Step 6: Mathematical Formula
    ax6 = plt.subplot(4, 2, (7, 8))
    ax6.axis('off')
    
    # Display the mathematical formula and real values
    formula_text = r'''$S_{perspective} = 0.30 \cdot C_{vanishing} + 0.25 \cdot C_{proportion} + 0.20 \cdot C_{parallel} + 0.15 \cdot V_{vertical} + 0.10 \cdot H_{horizontal}$

Real Calculation Example:
• $C_{vanishing}$ = ''' + f'{vanishing_consistency:.3f}' + r''' (Vanishing point consistency)
• $C_{proportion}$ = ''' + f'{proportion_consistency:.3f}' + r''' (Proportion consistency)
• $C_{parallel}$ = ''' + f'{parallel_consistency:.3f}' + r''' (Parallel line consistency)
• $V_{vertical}$ = ''' + f'{vertical_preservation:.3f}' + r''' (Vertical line preservation)
• $H_{horizontal}$ = ''' + f'{horizontal_preservation:.3f}' + r''' (Horizontal line preservation)

Final Score = ''' + f'{calculated_score:.3f}' + r'''

Algorithm: OpenCV (Canny + Hough Transform) + NumPy (Numerical Computation)'''
    
    ax6.text(0.05, 0.95, formula_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='serif',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax6.set_title('Step 6: Mathematical Formula with Real Values', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/perspective_analysis_real_data.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_analysis_real_data.pdf', bbox_inches='tight')
    print("✅ Real data perspective analysis image saved to: figures/perspective_analysis_real_data.png")
    print("✅ Real data perspective analysis image saved to: figures/perspective_analysis_real_data.pdf")
    
    plt.close()

def create_model_comparison_chart(best_examples):
    """Create a comparison chart of perspective scores across different models."""
    
    if len(best_examples) < 2:
        print("❌ Not enough examples for comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 1: Bar chart of perspective scores
    models = [ex['model'] for ex in best_examples]
    scores = [ex['perspective_score'] for ex in best_examples]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax1.bar(range(len(models)), scores, color=colors[:len(models)])
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.replace('-results', '') for m in models], rotation=45, ha='right')
    ax1.set_ylabel('Perspective Score')
    ax1.set_title('Top 5 Perspective Scores by Model', fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Scatter plot of different metrics
    if len(best_examples) >= 3:
        circulation_scores = []
        perspective_scores = []
        cot_scores = []
        model_labels = []
        
        for ex in best_examples:
            circulation_scores.append(ex['metrics'].get('overall_circulation_score', 0))
            perspective_scores.append(ex['perspective_score'])
            cot_scores.append(ex['metrics'].get('cot_score', 5) / 10.0)  # Normalize to 0-1
            model_labels.append(ex['model'].replace('-results', ''))
        
        # Create scatter plot
        scatter = ax2.scatter(circulation_scores, perspective_scores, 
                            s=[c*500 for c in cot_scores], 
                            c=range(len(model_labels)), 
                            cmap='tab10', alpha=0.7)
        
        # Add model labels
        for i, (x, y, label) in enumerate(zip(circulation_scores, perspective_scores, model_labels)):
            ax2.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Circulation Score')
        ax2.set_ylabel('Perspective Score')
        ax2.set_title('Circulation vs Perspective Scores\n(Bubble size = COT Score)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.0)
        ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('figures/perspective_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_model_comparison.pdf', bbox_inches='tight')
    print("✅ Model comparison chart saved to: figures/perspective_model_comparison.png")
    print("✅ Model comparison chart saved to: figures/perspective_model_comparison.pdf")
    
    plt.close()

def create_statistics_summary(all_results):
    """Create a statistics summary of perspective scores across all models."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_stats = {}
    all_scores = []
    
    # Collect all perspective scores by model
    for model, results in all_results.items():
        scores = []
        for item in results:
            if 'objective_metrics' in item and 'perspective_score' in item['objective_metrics']:
                score = item['objective_metrics']['perspective_score']
                scores.append(score)
                all_scores.append(score)
        
        if scores:
            model_stats[model] = {
                'scores': scores,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }
    
    # Chart 1: Box plot of scores by model
    ax1 = axes[0, 0]
    if model_stats:
        data_for_boxplot = [stats['scores'] for stats in model_stats.values()]
        labels = [model.replace('-results', '') for model in model_stats.keys()]
        
        bp = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_ylabel('Perspective Score')
        ax1.set_title('Perspective Score Distribution by Model', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
    
    # Chart 2: Histogram of all scores
    ax2 = axes[0, 1]
    if all_scores:
        ax2.hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Perspective Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Overall Perspective Score Distribution', fontweight='bold')
        ax2.axvline(np.mean(all_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_scores):.3f}')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # Chart 3: Mean scores by model
    ax3 = axes[1, 0]
    if model_stats:
        models = list(model_stats.keys())
        means = [model_stats[m]['mean'] for m in models]
        stds = [model_stats[m]['std'] for m in models]
        
        bars = ax3.bar(range(len(models)), means, yerr=stds, capsize=5, 
                      color=colors[:len(models)], alpha=0.7)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('-results', '') for m in models], rotation=45, ha='right')
        ax3.set_ylabel('Mean Perspective Score')
        ax3.set_title('Mean Perspective Score by Model (with std dev)', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Chart 4: Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if model_stats:
        # Create table data
        table_data = []
        headers = ['Model', 'Count', 'Mean', 'Std', 'Min', 'Max']
        
        for model, stats in model_stats.items():
            row = [
                model.replace('-results', ''),
                f"{stats['count']}",
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E8E8E8')
            else:
                cell.set_facecolor('#F8F8F8')
        
        ax4.set_title('Perspective Score Statistics Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('figures/perspective_statistics_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_statistics_summary.pdf', bbox_inches='tight')
    print("✅ Statistics summary saved to: figures/perspective_statistics_summary.png")
    print("✅ Statistics summary saved to: figures/perspective_statistics_summary.pdf")
    
    plt.close()

def main():
    print("🎨 Starting to generate perspective analysis images using real dataset results...")
    
    # Set matplotlib font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Load real data
    print("\n📊 Loading real evaluation results...")
    all_results = load_real_data()
    
    if not all_results:
        print("❌ No real data found!")
        return
    
    # Find best examples
    print("\n🔍 Finding best perspective examples...")
    best_examples = find_best_perspective_examples(all_results, top_k=10)
    
    if not best_examples:
        print("❌ No valid examples found!")
        return
    
    print(f"✅ Found {len(best_examples)} valid examples")
    for i, ex in enumerate(best_examples[:5]):
        print(f"  {i+1}. {ex['model']} - Image {ex['row_id']} - Score: {ex['perspective_score']:.4f}")
    
    # Generate visualizations
    print("\n🎨 Creating process visualization with real data...")
    create_real_data_process_visualization(best_examples)
    
    print("\n📊 Creating model comparison chart...")
    create_model_comparison_chart(best_examples[:5])
    
    print("\n📈 Creating statistics summary...")
    create_statistics_summary(all_results)
    
    print("\n🎉 All real data perspective analysis visualizations generated!")
    print("📁 Generated files:")
    print("   - figures/perspective_analysis_real_data.png")
    print("   - figures/perspective_analysis_real_data.pdf")
    print("   - figures/perspective_model_comparison.png")
    print("   - figures/perspective_model_comparison.pdf")
    print("   - figures/perspective_statistics_summary.png")
    print("   - figures/perspective_statistics_summary.pdf")

if __name__ == "__main__":
    main()
