#!/usr/bin/env python3
"""
Generate intermediate process images for perspective analysis algorithm
to help reviewers understand each step intuitively in the paper.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os
from perspective_analysis import PerspectiveAnalyzer

def create_step_by_step_images():
    """Create individual images for each step of the perspective analysis process."""
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Try to use a different real image from the dataset
    sample_paths = [
        "sora_image-results/images/41.png",  # High scoring example - different model
        "SD15-results/images/29.png",  # High scoring example
        "DALL-E-3/images/100.png",
        "gpt-image-1-results/images/50.png",
        "DALL-E-3/images/5.png"
    ]
    
    analyzer = PerspectiveAnalyzer()
    image = None
    
    for path in sample_paths:
        if os.path.exists(path):
            image = analyzer.load_image_from_path(path)
            if image is not None:
                print(f"✅ Using sample image: {path}")
                break
    
    if image is None:
        print("❌ No sample image found, creating synthetic example")
        image = create_synthetic_building_image()
    
    # Step 1: Original Image
    create_original_image_figure(image)
    
    # Step 2: Edge Detection
    create_edge_detection_figure(image, analyzer)
    
    # Step 3: Line Detection  
    create_line_detection_figure(image, analyzer)
    
    # Step 4: Vanishing Point Detection
    create_vanishing_point_figure(image, analyzer)
    
    # Step 5: Metrics Visualization
    create_metrics_visualization_figure(image, analyzer)
    
    # Create horizontal combined figure
    create_horizontal_combined_figure(image, analyzer)

def create_synthetic_building_image():
    """Create a synthetic building image if no real image is available."""
    # Create a 512x512 image
    image = np.ones((512, 512, 3), dtype=np.uint8) * 240
    
    # Draw a simple building with perspective
    # Building outline (trapezoid shape for perspective)
    points = np.array([[100, 450], [400, 450], [350, 150], [150, 150]], np.int32)
    cv2.fillPoly(image, [points], (200, 200, 200))
    cv2.polylines(image, [points], True, (0, 0, 0), 3)
    
    # Windows - create perspective effect
    for i in range(3):
        for j in range(2):
            # Calculate perspective-adjusted positions
            x_base = 160 + i * 60
            y_base = 200 + j * 80
            width = 40 - i * 2  # Slight perspective reduction
            height = 50
            
            # Window rectangle
            cv2.rectangle(image, (x_base, y_base), (x_base + width, y_base + height), (100, 150, 200), -1)
            cv2.rectangle(image, (x_base, y_base), (x_base + width, y_base + height), (0, 0, 0), 2)
    
    # Door
    cv2.rectangle(image, (230, 350), (270, 450), (139, 69, 19), -1)
    cv2.rectangle(image, (230, 350), (270, 450), (0, 0, 0), 2)
    
    # Roof line
    roof_points = np.array([[150, 150], [250, 100], [350, 150]], np.int32)
    cv2.fillPoly(image, [roof_points], (180, 180, 180))
    cv2.polylines(image, [roof_points], True, (0, 0, 0), 2)
    
    return image

def create_original_image_figure(image):
    """Create figure showing the original input image."""
    plt.figure(figsize=(8, 6))
    
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    plt.imshow(image_rgb)
    plt.title('Step 1: Original Architectural Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Add annotation
    plt.text(0.02, 0.98, f'Input: Generated architectural image\nResolution: {image.shape[1]}×{image.shape[0]}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/perspective_step1_original.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_step1_original.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Step 1 image saved: figures/perspective_step1_original.png")

def create_edge_detection_figure(image, analyzer):
    """Create figure showing edge detection results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image', fontweight='bold')
    ax1.axis('off')
    
    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Canny Edge Detection Result', fontweight='bold')
    ax2.axis('off')
    
    # Add parameters annotation
    ax2.text(0.02, 0.98, 'Canny Parameters:\n• Low threshold: 50\n• High threshold: 150\n• Gaussian blur: 5×5', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.suptitle('Step 2: Edge Detection using Canny Algorithm', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/perspective_step2_edges.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_step2_edges.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Step 2 image saved: figures/perspective_step2_edges.png")

def create_line_detection_figure(image, analyzer):
    """Create figure showing line detection results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Edge detection result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    ax1.imshow(edges, cmap='gray')
    ax1.set_title('Edge Detection Result', fontweight='bold')
    ax1.axis('off')
    
    # Line detection
    lines = analyzer.detect_lines(image)
    if len(image.shape) == 3:
        line_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    else:
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw detected lines
    for x1, y1, x2, y2 in lines[:50]:  # Limit for clarity
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    ax2.imshow(line_image)
    ax2.set_title('Hough Line Transform Result', fontweight='bold')
    ax2.axis('off')
    
    # Add detection stats
    ax2.text(0.02, 0.98, f'Hough Parameters:\n• Threshold: 100\n• Min line length: 50px\n• Max line gap: 10px\n\nDetected lines: {len(lines)}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle('Step 3: Line Detection using Hough Transform', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/perspective_step3_lines.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_step3_lines.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Step 3 image saved: figures/perspective_step3_lines.png")

def create_vanishing_point_figure(image, analyzer):
    """Create figure showing vanishing point detection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Line detection result
    lines = analyzer.detect_lines(image)
    if len(image.shape) == 3:
        line_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    else:
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for x1, y1, x2, y2 in lines[:50]:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    ax1.imshow(line_image)
    ax1.set_title('Detected Lines', fontweight='bold')
    ax1.axis('off')
    
    # Vanishing point detection
    vanishing_points = analyzer.detect_vanishing_points(lines, image.shape)
    if len(image.shape) == 3:
        vp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    else:
        vp_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Group lines by angle and draw with different colors
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
    
    # Draw grouped lines
    color_idx = 0
    for group_angle, group_lines in angle_groups.items():
        if len(group_lines) >= 2:
            color = colors[color_idx % len(colors)]
            for x1, y1, x2, y2 in group_lines:
                cv2.line(vp_image, (x1, y1), (x2, y2), color, 2)
            color_idx += 1
    
    # Draw vanishing points
    for vp_x, vp_y in vanishing_points:
        if 0 <= vp_x < image.shape[1] and 0 <= vp_y < image.shape[0]:
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 8, (255, 255, 255), -1)
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 8, (0, 0, 0), 2)
    
    ax2.imshow(vp_image)
    ax2.set_title('Vanishing Point Detection', fontweight='bold')
    ax2.axis('off')
    
    # Add vanishing point info
    parallel_groups = len([g for g in angle_groups.values() if len(g) >= 2])
    ax2.text(0.02, 0.98, f'Angle clustering:\n• Tolerance: 10°\n• Parallel groups: {parallel_groups}\n• Vanishing points: {len(vanishing_points)}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle('Step 4: Vanishing Point Detection via Angle Clustering', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/perspective_step4_vanishing.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_step4_vanishing.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Step 4 image saved: figures/perspective_step4_vanishing.png")

def create_metrics_visualization_figure(image, analyzer):
    """Create figure showing the five perspective metrics calculation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calculate all metrics
    lines = analyzer.detect_lines(image)
    vanishing_points = analyzer.detect_vanishing_points(lines, image.shape)
    
    vanishing_consistency = analyzer.calculate_vanishing_point_consistency(lines, vanishing_points)
    proportion_consistency = analyzer.calculate_proportion_consistency(lines)
    parallel_consistency = analyzer.calculate_parallel_line_consistency(lines)
    vertical_preservation = analyzer.calculate_vertical_line_preservation(lines)
    horizontal_preservation = analyzer.calculate_horizontal_line_preservation(lines)
    
    # Metric 1: Vanishing Point Consistency
    ax1.set_title('Vanishing Point Consistency', fontweight='bold')
    
    # Visualize distance from lines to vanishing points
    if len(image.shape) == 3:
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    else:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Draw lines with colors based on distance to nearest vanishing point
    for x1, y1, x2, y2 in lines[:30]:
        # Calculate distance to nearest vanishing point
        min_dist = float('inf')
        for vp_x, vp_y in vanishing_points:
            # Calculate distance from vanishing point to line
            # Using perpendicular distance formula
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A != 0 or B != 0:
                dist = abs(A * vp_x + B * vp_y + C) / np.sqrt(A**2 + B**2)
            else:
                dist = float('inf')
            min_dist = min(min_dist, dist)
        
        # Color based on distance (closer = greener, farther = redder)
        if min_dist < 50:
            color = (0, 255, 0)  # Green - good consistency
        elif min_dist < 100:
            color = (255, 255, 0)  # Yellow - moderate consistency
        else:
            color = (255, 0, 0)  # Red - poor consistency
        
        cv2.line(vis_image, (x1, y1), (x2, y2), color, 2)
    
    ax1.imshow(vis_image)
    ax1.axis('off')
    ax1.text(0.02, 0.98, f'Score: {vanishing_consistency:.3f}\nGreen: Good convergence\nYellow: Moderate\nRed: Poor', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Metric 2: Proportion Consistency
    ax2.set_title('Proportion Consistency', fontweight='bold')
    
    # Calculate line lengths and ratios
    line_lengths = []
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        line_lengths.append(length)
    
    if line_lengths:
        ax2.hist(line_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Line Length (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.text(0.02, 0.98, f'Score: {proportion_consistency:.3f}\nAnalyzes length ratios\nvs architectural standards', 
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Metric 3: Parallel Line Consistency  
    ax3.set_title('Parallel Line Consistency', fontweight='bold')
    
    # Group lines by angle and show angle distribution
    angles = []
    for x1, y1, x2, y2 in lines:
        angle = analyzer.calculate_line_angle(x1, y1, x2, y2)
        angles.append(angle * 180 / np.pi)  # Convert to degrees
    
    if angles:
        ax3.hist(angles, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('Line Angle (degrees)')
        ax3.set_ylabel('Frequency')
        ax3.text(0.02, 0.98, f'Score: {parallel_consistency:.3f}\nMeasures angle uniformity\nwithin parallel groups', 
                 transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Metric 4 & 5: Vertical and Horizontal Preservation
    ax4.set_title('Vertical & Horizontal Preservation', fontweight='bold')
    
    # Show vertical and horizontal lines
    vertical_lines = 0
    horizontal_lines = 0
    
    if len(image.shape) == 3:
        vh_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    else:
        vh_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for x1, y1, x2, y2 in lines:
        angle = analyzer.calculate_line_angle(x1, y1, x2, y2)
        angle_deg = abs(angle * 180 / np.pi)
        
        # Check if line is close to vertical (90 degrees)
        if abs(angle_deg - 90) < 15:
            cv2.line(vh_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for vertical
            vertical_lines += 1
        # Check if line is close to horizontal (0 or 180 degrees)
        elif angle_deg < 15 or angle_deg > 165:
            cv2.line(vh_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red for horizontal
            horizontal_lines += 1
    
    ax4.imshow(vh_image)
    ax4.axis('off')
    ax4.text(0.02, 0.98, f'Vertical Score: {vertical_preservation:.3f}\nHorizontal Score: {horizontal_preservation:.3f}\n\nGreen: Vertical lines\nRed: Horizontal lines', 
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.suptitle('Step 5: Five Perspective Consistency Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/perspective_step5_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_step5_metrics.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Step 5 image saved: figures/perspective_step5_metrics.png")

def create_horizontal_combined_figure(image, analyzer):
    """Create a horizontal layout showing all 5 steps in one figure."""
    
    # Create a wide figure for horizontal layout
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Calculate all metrics first
    lines = analyzer.detect_lines(image)
    vanishing_points = analyzer.detect_vanishing_points(lines, image.shape)
    
    vanishing_consistency = analyzer.calculate_vanishing_point_consistency(lines, vanishing_points)
    proportion_consistency = analyzer.calculate_proportion_consistency(lines)
    parallel_consistency = analyzer.calculate_parallel_line_consistency(lines)
    vertical_preservation = analyzer.calculate_vertical_line_preservation(lines)
    horizontal_preservation = analyzer.calculate_horizontal_line_preservation(lines)
    
    # Step 1: Original Image
    ax1 = axes[0, 0]
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    ax1.imshow(image_rgb)
    ax1.set_title('Step 1: Original\nArchitectural Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Step 2: Edge Detection
    ax2 = axes[0, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Step 2: Canny\nEdge Detection', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Step 3: Line Detection
    ax3 = axes[0, 2]
    line_image = image_rgb.copy()
    for x1, y1, x2, y2 in lines[:50]:  # Limit for clarity
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    ax3.imshow(line_image)
    ax3.set_title('Step 3: Hough\nLine Detection', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Step 4: Vanishing Point Detection
    ax4 = axes[0, 3]
    vp_image = image_rgb.copy()
    
    # Group lines by angle and draw with different colors
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
    
    # Draw grouped lines
    color_idx = 0
    for group_angle, group_lines in angle_groups.items():
        if len(group_lines) >= 2:
            color = colors[color_idx % len(colors)]
            for x1, y1, x2, y2 in group_lines:
                cv2.line(vp_image, (x1, y1), (x2, y2), color, 2)
            color_idx += 1
    
    # Draw vanishing points
    for vp_x, vp_y in vanishing_points:
        if 0 <= vp_x < image.shape[1] and 0 <= vp_y < image.shape[0]:
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 8, (255, 255, 255), -1)
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 8, (0, 0, 0), 2)
    
    ax4.imshow(vp_image)
    ax4.set_title('Step 4: Vanishing\nPoint Detection', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Step 5: Metrics Calculation (Bar Chart)
    ax5 = axes[0, 4]
    
    metrics = {
        'Vanishing\nPoint': vanishing_consistency,
        'Proportion': proportion_consistency,
        'Parallel\nLine': parallel_consistency,
        'Vertical\nLine': vertical_preservation,
        'Horizontal\nLine': horizontal_preservation
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax5.bar(range(len(metric_names)), metric_values, color=colors_bar)
    ax5.set_xticks(range(len(metric_names)))
    ax5.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Score', fontsize=10)
    ax5.set_title('Step 5: Perspective\nConsistency Metrics', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1.0)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value, weight) in enumerate(zip(bars, metric_values, weights)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.2f}\n({weight:.0%})', ha='center', va='bottom', fontsize=8)
    
    # Bottom row: Detailed parameter information for each step
    
    # Step 1 details
    ax1_detail = axes[1, 0]
    ax1_detail.axis('off')
    ax1_detail.text(0.5, 0.5, f'Input Image\n\nResolution:\n{image.shape[1]}×{image.shape[0]}\n\nSource:\nReal dataset example', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Step 2 details
    ax2_detail = axes[1, 1]
    ax2_detail.axis('off')
    ax2_detail.text(0.5, 0.5, 'Canny Parameters:\n\n• Low threshold: 50\n• High threshold: 150\n• Gaussian blur: 5×5\n• Kernel size: (5,5)', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Step 3 details
    ax3_detail = axes[1, 2]
    ax3_detail.axis('off')
    ax3_detail.text(0.5, 0.5, f'Hough Parameters:\n\n• Threshold: 100\n• Min line length: 50px\n• Max line gap: 10px\n• Lines detected: {len(lines)}', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Step 4 details
    ax4_detail = axes[1, 3]
    ax4_detail.axis('off')
    parallel_groups = len([g for g in angle_groups.values() if len(g) >= 2])
    ax4_detail.text(0.5, 0.5, f'Clustering Parameters:\n\n• Angle tolerance: 10°\n• Parallel groups: {parallel_groups}\n• Vanishing points: {len(vanishing_points)}\n• Method: RANSAC-like', 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    # Step 5 details (Final Score)
    ax5_detail = axes[1, 4]
    ax5_detail.axis('off')
    
    # Calculate final score
    final_score = sum(v * w for v, w in zip(metric_values, weights))
    
    formula_text = f'Final Score Calculation:\n\nS = 0.30×{vanishing_consistency:.2f}\n  + 0.25×{proportion_consistency:.2f}\n  + 0.20×{parallel_consistency:.2f}\n  + 0.15×{vertical_preservation:.2f}\n  + 0.10×{horizontal_preservation:.2f}\n\nFinal Score = {final_score:.3f}'
    
    ax5_detail.text(0.5, 0.5, formula_text, 
                   ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8))
    
    plt.suptitle('Perspective Consistency Analysis: Five-Step Algorithm Process', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.savefig('figures/perspective_horizontal_process.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_horizontal_process.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Horizontal combined image saved: figures/perspective_horizontal_process.png")

def main():
    print("🎨 Generating intermediate process images for perspective analysis...")
    
    # Set matplotlib parameters
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Generate all step-by-step images
    create_step_by_step_images()
    
    print("\n🎉 All perspective analysis process step images generated!")
    print("📁 Generated files:")
    print("   - figures/perspective_step1_original.png")
    print("   - figures/perspective_step2_edges.png") 
    print("   - figures/perspective_step3_lines.png")
    print("   - figures/perspective_step4_vanishing.png")
    print("   - figures/perspective_step5_metrics.png")
    print("   - figures/perspective_horizontal_process.png (NEW: All steps in one horizontal layout)")
    print("\n💡 The horizontal layout image is perfect for your LaTeX paper to show the complete algorithm flow!")

if __name__ == "__main__":
    main()
