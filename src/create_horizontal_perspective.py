#!/usr/bin/env python3
"""
Create a horizontal layout showing all 5 perspective analysis steps in one figure.
Uses a different example image as requested.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from perspective_analysis import PerspectiveAnalyzer

def main():
    print("🎨 Creating horizontal perspective analysis process image...")
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Use DALL-E-3/images/1.png as requested
    sample_paths = [
        "DALL-E-3/images/1.png",  # Requested example
        "DALL-E-3/images/5.png",
        "DALL-E-3/images/10.png",
        "sora_image-results/images/41.png",
        "SD15-results/images/29.png"
    ]
    
    analyzer = PerspectiveAnalyzer()
    image = None
    used_path = None
    
    for path in sample_paths:
        if os.path.exists(path):
            image = analyzer.load_image_from_path(path)
            if image is not None:
                used_path = path
                print(f"✅ Using sample image: {path}")
                break
    
    if image is None:
        print("❌ No sample image found, creating synthetic example")
        image = create_synthetic_building_image()
        used_path = "synthetic"
    
    # Create horizontal layout
    create_horizontal_combined_figure(image, analyzer, used_path)
    
    print("\n🎉 Horizontal perspective analysis image generated!")
    print("📁 Generated file: figures/perspective_horizontal_process.png")
    print("💡 This image shows all 5 algorithm steps in one horizontal layout!")

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

def create_horizontal_combined_figure(image, analyzer, source_path):
    """Create an optimized horizontal layout showing all 5 steps in one figure."""
    
    # Set matplotlib parameters for better rendering
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # Create a wide figure for horizontal layout - single row design with lower height
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Calculate all metrics first
    lines = analyzer.detect_lines(image)
    vanishing_points = analyzer.detect_vanishing_points(lines, image.shape)
    
    vanishing_consistency = analyzer.calculate_vanishing_point_consistency(lines, vanishing_points)
    proportion_consistency = analyzer.calculate_proportion_consistency(lines)
    parallel_consistency = analyzer.calculate_parallel_line_consistency(lines)
    vertical_preservation = analyzer.calculate_vertical_line_preservation(lines)
    horizontal_preservation = analyzer.calculate_horizontal_line_preservation(lines)
    
    print(f"📊 Calculated metrics for {source_path}:")
    print(f"   - Vanishing point consistency: {vanishing_consistency:.3f}")
    print(f"   - Proportion consistency: {proportion_consistency:.3f}")
    print(f"   - Parallel line consistency: {parallel_consistency:.3f}")
    print(f"   - Vertical preservation: {vertical_preservation:.3f}")
    print(f"   - Horizontal preservation: {horizontal_preservation:.3f}")
    
    # Step 1: Original Image
    ax1 = axes[0]
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    ax1.imshow(image_rgb)
    ax1.set_title('Step 1: Original Architectural Image', fontsize=14, fontweight='bold', pad=20)
    ax1.axis('off')
    ax1.set_aspect('equal')  # Force same aspect ratio
    
    # Add data annotation directly on the image
    source_info = source_path.split('/')[-1] if '/' in source_path else source_path
    ax1.text(0.02, 0.02, f'Source: {source_info}\nResolution: {image.shape[1]}×{image.shape[0]}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='gray'))
    
    # Step 2: Edge Detection
    ax2 = axes[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Step 2: Canny Edge Detection', fontsize=14, fontweight='bold', pad=20)
    ax2.axis('off')
    ax2.set_aspect('equal')  # Force same aspect ratio
    
    # Add parameters annotation directly on the image
    ax2.text(0.02, 0.98, 'Parameters:\n• Low threshold: 50\n• High threshold: 150\n• Gaussian blur: 5×5', 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9, edgecolor='orange'))
    
    # Step 3: Line Detection
    ax3 = axes[2]
    line_image = image_rgb.copy()
    for x1, y1, x2, y2 in lines[:50]:  # Limit for clarity
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    ax3.imshow(line_image)
    ax3.set_title('Step 3: Hough Line Detection', fontsize=14, fontweight='bold', pad=20)
    ax3.axis('off')
    ax3.set_aspect('equal')  # Force same aspect ratio
    
    # Add detection stats directly on the image
    ax3.text(0.02, 0.98, f'Parameters:\n• Threshold: 100\n• Min length: 50px\n• Max gap: 10px\n\nDetected: {len(lines)} lines', 
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9, edgecolor='green'))
    
    # Step 4: Vanishing Point Detection
    ax4 = axes[3]
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
    
    # Draw grouped lines with thicker lines
    color_idx = 0
    for group_angle, group_lines in angle_groups.items():
        if len(group_lines) >= 2:
            color = colors[color_idx % len(colors)]
            for x1, y1, x2, y2 in group_lines:
                cv2.line(vp_image, (x1, y1), (x2, y2), color, 3)
            color_idx += 1
    
    # Draw vanishing points with better visibility
    for vp_x, vp_y in vanishing_points:
        if 0 <= vp_x < image.shape[1] and 0 <= vp_y < image.shape[0]:
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 12, (255, 255, 255), -1)
            cv2.circle(vp_image, (int(vp_x), int(vp_y)), 12, (0, 0, 0), 3)
    
    ax4.imshow(vp_image)
    ax4.set_title('Step 4: Vanishing Point Detection', fontsize=14, fontweight='bold', pad=20)
    ax4.axis('off')
    ax4.set_aspect('equal')  # Force same aspect ratio
    
    # Add vanishing point info directly on the image
    parallel_groups = len([g for g in angle_groups.values() if len(g) >= 2])
    ax4.text(0.02, 0.98, f'Clustering:\n• Tolerance: 10°\n• Parallel groups: {parallel_groups}\n• Vanishing points: {len(vanishing_points)}', 
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9, edgecolor='blue'))
    
    # Step 5: Metrics Visualization (forced to same size as other steps)
    ax5 = axes[4]
    
    # Create a beautiful donut chart with size constraints
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
    
    # Beautiful gradient colors
    colors_gradient = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Force the axis to have the same aspect ratio as image axes
    ax5.set_aspect('equal')
    
    # Create a fancy, modern donut chart with gradient effects
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    
    # Enhanced color palette with gradients
    fancy_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    shadow_colors = ['#E55555', '#3BB8B3', '#3A9FC0', '#85BA9D', '#FFDD8B']
    
    # Create multiple ring layers for depth effect
    outer_radius = 0.5
    inner_radius = 0.32
    shadow_offset = 0.02
    
    # Add shadow layer first
    shadow_wedges, _, _ = ax5.pie(weights, labels=None, colors=shadow_colors,
                                 autopct='', startangle=90,
                                 wedgeprops=dict(width=outer_radius-inner_radius-0.01, 
                                               edgecolor='none'),
                                 radius=outer_radius-shadow_offset,
                                 center=(shadow_offset, -shadow_offset))
    
    # Main donut chart with enhanced styling
    wedges, texts, autotexts = ax5.pie(weights, labels=None, colors=fancy_colors, 
                                      autopct='', startangle=90, 
                                      wedgeprops=dict(width=outer_radius-inner_radius, 
                                                    edgecolor='white', linewidth=3),
                                      radius=outer_radius)
    
    # Add gradient effect to each wedge
    for i, wedge in enumerate(wedges):
        # Add inner glow effect
        inner_wedge = mpatches.Wedge(wedge.center, inner_radius + 0.04, 
                                   wedge.theta1, wedge.theta2,
                                   facecolor=fancy_colors[i], alpha=0.3)
        ax5.add_patch(inner_wedge)
    
    # Stylish center circle with gradient border
    centre_circle = plt.Circle((0,0), inner_radius, fc='white', 
                              edgecolor='#DDDDDD', linewidth=2)
    ax5.add_artist(centre_circle)
    
    # Add inner decorative ring
    inner_ring = plt.Circle((0,0), inner_radius-0.02, fc='none', 
                           edgecolor='#F0F0F0', linewidth=1)
    ax5.add_artist(inner_ring)
    
    # Calculate final score
    final_score = sum(v * w for v, w in zip(metric_values, weights))
    
    # Fancy center text with better typography
    ax5.text(0, 0.05, 'Final Score', ha='center', va='center', 
             fontsize=8, fontweight='600', color='#666666')
    ax5.text(0, -0.05, f'{final_score:.3f}', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#2C3E50')
    
    # Add decorative elements around the center
    for angle in [0, 90, 180, 270]:
        x = 0.15 * np.cos(np.radians(angle))
        y = 0.15 * np.sin(np.radians(angle))
        dot = plt.Circle((x, y), 0.015, fc='#BDC3C7', alpha=0.7)
        ax5.add_artist(dot)
    
    # Fancy floating labels outside the ring
    label_radius = outer_radius + 0.15
    for i, (wedge, name, value, weight) in enumerate(zip(wedges, metric_names, metric_values, weights)):
        # Calculate angle for text placement
        angle = (wedge.theta1 + wedge.theta2) / 2
        x = label_radius * np.cos(np.radians(angle))
        y = label_radius * np.sin(np.radians(angle))
        
        # Simplified names
        short_names = ['VP', 'PR', 'PL', 'VT', 'HZ']
        short_name = short_names[i]
        
        # Create fancy floating label with shadow
        # Shadow
        ax5.text(x+0.01, y-0.01, f'{short_name}\n{value:.2f}', ha='center', va='center', 
                fontsize=7, fontweight='bold', color='black', alpha=0.3)
        
        # Main label with modern styling
        ax5.text(x, y, f'{short_name}\n{value:.2f}', ha='center', va='center', 
                fontsize=7, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.1", facecolor=fancy_colors[i], 
                         edgecolor='white', linewidth=2, alpha=0.95))
        
        # Add connecting line from wedge to label
        wedge_radius = (outer_radius + inner_radius) / 2
        wedge_x = wedge_radius * np.cos(np.radians(angle))
        wedge_y = wedge_radius * np.sin(np.radians(angle))
        
        # Stylish connecting line
        ax5.plot([wedge_x, x*0.85], [wedge_y, y*0.85], 
                color=fancy_colors[i], linewidth=1.5, alpha=0.6)
    
    # Add subtle background pattern
    bg_circle = plt.Circle((0,0), outer_radius + 0.3, fc='none', 
                          edgecolor='#F8F9FA', linewidth=0.5, alpha=0.5)
    ax5.add_artist(bg_circle)
    
    # Set limits with some padding
    ax5.set_xlim(-0.9, 0.9)
    ax5.set_ylim(-0.9, 0.9)
    
    ax5.set_title('Step 5: Perspective Consistency Metrics', fontsize=14, fontweight='bold', pad=20)
    
    # No overall title as requested
    # Adjust layout for better spacing and ensure title alignment
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    
    # Force all titles to be at exactly the same height
    # Get the maximum y position of all axes to align titles
    max_y = max([ax.get_position().y1 for ax in axes])
    title_y_position = max_y + 0.08  # Fixed position above all axes (adjusted for lower height)
    
    # Set all titles at the exact same y position
    for i, ax in enumerate(axes):
        ax_pos = ax.get_position()
        title_x_position = ax_pos.x0 + ax_pos.width / 2  # Center of each subplot
        
        # Remove existing title
        ax.set_title('')
        
        # Add title at fixed position
        titles = [
            'Step 1: Original Architectural Image',
            'Step 2: Canny Edge Detection', 
            'Step 3: Hough Line Detection',
            'Step 4: Vanishing Point Detection',
            'Step 5: Perspective Consistency Metrics'
        ]
        
        fig.text(title_x_position, title_y_position, titles[i], 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.savefig('figures/perspective_horizontal_process.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/perspective_horizontal_process.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Horizontal combined image saved: figures/perspective_horizontal_process.png")
    print("✅ Horizontal combined image saved: figures/perspective_horizontal_process.pdf")

if __name__ == "__main__":
    main()
