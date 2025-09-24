#!/usr/bin/env python3
"""
Perspective Analysis Module for Architecture Evaluation
Implements perspective geometry consistency assessment based on computer vision techniques

This module implements perspective rationality analysis using:
- OpenCV for line detection and Hough transform
- NumPy for geometric calculations
- Mathematical perspective geometry principles

Author: Architecture Benchmark System
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import defaultdict
import os
from PIL import Image
import requests
from io import BytesIO


class PerspectiveAnalyzer:
    """
    Main class for perspective analysis using computer vision algorithms.
    Implements all the mathematical formulas from the paper.
    """
    
    def __init__(self):
        self.line_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 100,
            'min_line_length': 50,
            'max_line_gap': 10
        }
        
        # Perspective evaluation weights
        self.weights = {
            'vanishing_point_consistency': 0.3,
            'proportion_consistency': 0.25,
            'parallel_line_consistency': 0.2,
            'vertical_line_preservation': 0.15,
            'horizontal_line_preservation': 0.1
        }
    
    def load_image_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from local file path or URL.
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                # Load from local file
                image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not load image from {image_path}")
                return None
            
            return image
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            return None
    
    def detect_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect lines in image using Canny edge detection and Hough transform.
        Returns list of lines as (x1, y1, x2, y2) tuples.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 
                         self.line_detection_params['canny_low'], 
                         self.line_detection_params['canny_high'])
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges,
                               rho=1,
                               theta=np.pi/180,
                               threshold=self.line_detection_params['hough_threshold'],
                               minLineLength=self.line_detection_params['min_line_length'],
                               maxLineGap=self.line_detection_params['max_line_gap'])
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append((x1, y1, x2, y2))
        
        return line_list
    
    def calculate_line_angle(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Calculate angle of line in radians.
        """
        return math.atan2(y2 - y1, x2 - x1)
    
    def line_intersection(self, line1: Tuple[int, int, int, int], 
                         line2: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
        """
        Calculate intersection point of two lines.
        Returns None if lines are parallel.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        
        return (px, py)
    
    def detect_vanishing_points(self, lines: List[Tuple[int, int, int, int]], 
                               image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """
        Detect vanishing points using RANSAC-like approach.
        Groups parallel lines and finds their intersection points.
        """
        if len(lines) < 2:
            return []
        
        # Group lines by angle (with tolerance)
        angle_tolerance = np.pi / 18  # 10 degrees
        line_groups = defaultdict(list)
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = self.calculate_line_angle(x1, y1, x2, y2)
            
            # Normalize angle to [0, pi)
            if angle < 0:
                angle += np.pi
            
            # Find existing group or create new one
            assigned = False
            for group_angle in line_groups.keys():
                if abs(angle - group_angle) < angle_tolerance:
                    line_groups[group_angle].append(line)
                    assigned = True
                    break
            
            if not assigned:
                line_groups[angle] = [line]
        
        # Find vanishing points for each group
        vanishing_points = []
        height, width = image_shape[:2]
        
        for angle, group_lines in line_groups.items():
            if len(group_lines) < 2:
                continue
            
            # Find intersections between lines in this group
            intersections = []
            for i in range(len(group_lines)):
                for j in range(i + 1, len(group_lines)):
                    intersection = self.line_intersection(group_lines[i], group_lines[j])
                    if intersection is not None:
                        px, py = intersection
                        # Filter out points too far from image
                        if (-width < px < 2*width) and (-height < py < 2*height):
                            intersections.append((px, py))
            
            if intersections:
                # Use median as robust estimate of vanishing point
                px_values = [p[0] for p in intersections]
                py_values = [p[1] for p in intersections]
                vp_x = np.median(px_values)
                vp_y = np.median(py_values)
                vanishing_points.append((vp_x, vp_y))
        
        return vanishing_points
    
    def calculate_vanishing_point_consistency(self, lines: List[Tuple[int, int, int, int]], 
                                            vanishing_points: List[Tuple[float, float]]) -> float:
        """
        Calculate vanishing point consistency score.
        C_vanishing = (1/|L|) * Σ exp(-d(l, VP_nearest)^2 / 2σ^2)
        """
        if not lines or not vanishing_points:
            return 0.5  # Default score
        
        sigma = 100.0  # Distance tolerance parameter
        total_score = 0.0
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Find nearest vanishing point
            min_distance = float('inf')
            for vp_x, vp_y in vanishing_points:
                # Calculate distance from line to vanishing point
                # Using point-to-line distance formula
                A = y2 - y1
                B = x1 - x2
                C = x2*y1 - x1*y2
                
                if A == 0 and B == 0:  # Degenerate line
                    continue
                
                distance = abs(A*vp_x + B*vp_y + C) / math.sqrt(A*A + B*B)
                min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                score = math.exp(-(min_distance**2) / (2 * sigma**2))
                total_score += score
        
        return total_score / len(lines)
    
    def calculate_proportion_consistency(self, lines: List[Tuple[int, int, int, int]]) -> float:
        """
        Calculate proportion consistency based on line length ratios.
        C_proportion = 1 - (1/|R|) * Σ |S_r^observed / S_r^expected - 1|
        """
        if len(lines) < 2:
            return 1.0
        
        # Calculate line lengths
        lengths = []
        for x1, y1, x2, y2 in lines:
            length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            lengths.append(length)
        
        lengths.sort()
        
        # Check ratios between consecutive lengths
        if len(lengths) < 2:
            return 1.0
        
        # Expected ratios based on perspective theory (simplified)
        # In perspective, parallel lines should maintain certain proportional relationships
        total_deviation = 0.0
        ratio_count = 0
        
        for i in range(len(lengths) - 1):
            if lengths[i] > 0:
                observed_ratio = lengths[i+1] / lengths[i]
                # Expected ratio around golden ratio or simple ratios
                expected_ratios = [1.0, 1.618, 2.0, 0.618, 0.5]  # Common architectural ratios
                
                min_deviation = min(abs(observed_ratio - expected) for expected in expected_ratios)
                total_deviation += min_deviation
                ratio_count += 1
        
        if ratio_count == 0:
            return 1.0
        
        average_deviation = total_deviation / ratio_count
        return max(0.0, 1.0 - average_deviation)
    
    def calculate_parallel_line_consistency(self, lines: List[Tuple[int, int, int, int]]) -> float:
        """
        Calculate parallel line consistency.
        Lines that should be parallel should have similar angles.
        """
        if len(lines) < 2:
            return 1.0
        
        # Group lines by angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = self.calculate_line_angle(x1, y1, x2, y2)
            # Normalize to [0, pi)
            if angle < 0:
                angle += np.pi
            angles.append(angle)
        
        # Find groups of parallel lines
        angle_tolerance = np.pi / 18  # 10 degrees
        parallel_groups = []
        used = set()
        
        for i, angle1 in enumerate(angles):
            if i in used:
                continue
            
            group = [i]
            used.add(i)
            
            for j, angle2 in enumerate(angles):
                if j in used:
                    continue
                
                if abs(angle1 - angle2) < angle_tolerance or abs(angle1 - angle2 - np.pi) < angle_tolerance:
                    group.append(j)
                    used.add(j)
            
            if len(group) > 1:
                parallel_groups.append(group)
        
        if not parallel_groups:
            return 0.5  # No clear parallel groups found
        
        # Calculate consistency within each group
        total_consistency = 0.0
        total_lines = 0
        
        for group in parallel_groups:
            if len(group) < 2:
                continue
            
            group_angles = [angles[i] for i in group]
            mean_angle = np.mean(group_angles)
            
            # Calculate deviation from mean
            deviations = [abs(angle - mean_angle) for angle in group_angles]
            group_consistency = 1.0 - np.mean(deviations) / (np.pi / 4)  # Normalize by 45 degrees
            
            total_consistency += group_consistency * len(group)
            total_lines += len(group)
        
        return max(0.0, total_consistency / total_lines if total_lines > 0 else 0.5)
    
    def calculate_vertical_line_preservation(self, lines: List[Tuple[int, int, int, int]]) -> float:
        """
        Calculate vertical line preservation score.
        V_vertical = (1/|L_v|) * Σ cos(θ_l)
        """
        if not lines:
            return 1.0
        
        vertical_threshold = np.pi / 12  # 15 degrees tolerance
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = abs(self.calculate_line_angle(x1, y1, x2, y2))
            
            # Check if line is close to vertical (angle close to π/2)
            vertical_deviation = abs(angle - np.pi/2)
            if vertical_deviation < vertical_threshold:
                vertical_lines.append(angle)
        
        if not vertical_lines:
            return 0.5  # No clear vertical lines found
        
        # Calculate how close to vertical these lines are
        total_score = 0.0
        for angle in vertical_lines:
            deviation = abs(angle - np.pi/2)
            score = math.cos(deviation)  # cos(0) = 1 for perfect vertical
            total_score += score
        
        return total_score / len(vertical_lines)
    
    def calculate_horizontal_line_preservation(self, lines: List[Tuple[int, int, int, int]]) -> float:
        """
        Calculate horizontal line preservation score.
        H_horizontal = (1/|L_h|) * Σ cos(φ_l)
        """
        if not lines:
            return 1.0
        
        horizontal_threshold = np.pi / 12  # 15 degrees tolerance
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = abs(self.calculate_line_angle(x1, y1, x2, y2))
            
            # Check if line is close to horizontal (angle close to 0 or π)
            horizontal_deviation = min(abs(angle), abs(angle - np.pi))
            if horizontal_deviation < horizontal_threshold:
                horizontal_lines.append(angle)
        
        if not horizontal_lines:
            return 0.5  # No clear horizontal lines found
        
        # Calculate how close to horizontal these lines are
        total_score = 0.0
        for angle in horizontal_lines:
            deviation = min(abs(angle), abs(angle - np.pi))
            score = math.cos(deviation)  # cos(0) = 1 for perfect horizontal
            total_score += score
        
        return total_score / len(horizontal_lines)
    
    def calculate_overall_perspective_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall perspective score using weighted combination.
        PCS = α·C_vanishing + β·C_proportion + γ·C_parallel + δ·V_vertical + ε·H_horizontal
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in self.weights.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def analyze_perspective(self, image_path: str) -> Dict[str, Any]:
        """
        Main function to analyze perspective for a given image.
        """
        try:
            # Step 1: Load image
            image = self.load_image_from_path(image_path)
            if image is None:
                return self._get_default_result("Image loading failed")
            
            # Step 2: Detect lines
            lines = self.detect_lines(image)
            if not lines:
                return self._get_default_result("No lines detected")
            
            # Step 3: Detect vanishing points
            vanishing_points = self.detect_vanishing_points(lines, image.shape)
            
            # Step 4: Calculate all perspective metrics
            metrics = {}
            
            # Vanishing point consistency
            metrics['vanishing_point_consistency'] = self.calculate_vanishing_point_consistency(
                lines, vanishing_points)
            
            # Proportion consistency
            metrics['proportion_consistency'] = self.calculate_proportion_consistency(lines)
            
            # Parallel line consistency
            metrics['parallel_line_consistency'] = self.calculate_parallel_line_consistency(lines)
            
            # Vertical line preservation
            metrics['vertical_line_preservation'] = self.calculate_vertical_line_preservation(lines)
            
            # Horizontal line preservation
            metrics['horizontal_line_preservation'] = self.calculate_horizontal_line_preservation(lines)
            
            # Overall perspective score
            overall_score = self.calculate_overall_perspective_score(metrics)
            
            return {
                'perspective_score': overall_score,
                'detailed_metrics': metrics,
                'analysis_stats': {
                    'lines_detected': len(lines),
                    'vanishing_points_found': len(vanishing_points),
                    'image_shape': image.shape[:2]
                }
            }
            
        except Exception as e:
            print(f"Warning: Perspective analysis failed: {e}")
            return self._get_default_result(f"Analysis error: {e}")
    
    def _get_default_result(self, reason: str = "") -> Dict[str, Any]:
        """
        Return default result when analysis fails.
        """
        return {
            'perspective_score': 0.5,
            'detailed_metrics': {
                'vanishing_point_consistency': 0.5,
                'proportion_consistency': 0.5,
                'parallel_line_consistency': 0.5,
                'vertical_line_preservation': 0.5,
                'horizontal_line_preservation': 0.5
            },
            'analysis_stats': {
                'lines_detected': 0,
                'vanishing_points_found': 0,
                'image_shape': (0, 0),
                'failure_reason': reason
            }
        }


def test_perspective_analyzer():
    """Test function for the perspective analyzer."""
    analyzer = PerspectiveAnalyzer()
    
    # Test with a sample image (if available)
    test_image_paths = [
        "DALL-E-3/images/1.png",
        "gpt-image-1-results/images/1.png",
        "SD15-results/images/1.png"
    ]
    
    for image_path in test_image_paths:
        if os.path.exists(image_path):
            print(f"\n=== Testing with {image_path} ===")
            result = analyzer.analyze_perspective(image_path)
            
            print(f"Overall perspective score: {result['perspective_score']:.3f}")
            print("Detailed metrics:")
            for key, value in result['detailed_metrics'].items():
                print(f"  {key}: {value:.3f}")
            print(f"Analysis stats: {result['analysis_stats']}")
            break
    else:
        print("No test images found. Creating synthetic test...")
        # Test with default values
        result = analyzer._get_default_result("No test images available")
        print(f"Default result: {result}")


if __name__ == "__main__":
    test_perspective_analyzer()
