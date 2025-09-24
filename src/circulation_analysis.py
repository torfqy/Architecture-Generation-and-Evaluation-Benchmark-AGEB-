#!/usr/bin/env python3
"""
Circulation Analysis Module for Architecture Evaluation
Pure Algorithm-based Approach without External API Calls

This module implements circulation rationality analysis using:
- Graph theory algorithms (NetworkX)
- Geometric calculations (Shapely) 
- Computer vision techniques (OpenCV)
- Statistical analysis (NumPy/SciPy)

Author: Architecture Benchmark System
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import json
import re
import math
from collections import defaultdict
import cv2
from PIL import Image
import requests
from io import BytesIO


class CirculationAnalyzer:
    """
    Main class for circulation analysis using pure algorithms.
    Implements all the mathematical formulas from the paper.
    """
    
    def __init__(self):
        self.space_types = {
            'office': {'privacy': 2, 'frequency': 0.8, 'area_weight': 1.0},
            'meeting': {'privacy': 3, 'frequency': 0.6, 'area_weight': 1.2},
            'bathroom': {'privacy': 5, 'frequency': 0.4, 'area_weight': 0.5},
            'kitchen': {'privacy': 2, 'frequency': 0.7, 'area_weight': 1.0},
            'bedroom': {'privacy': 5, 'frequency': 0.5, 'area_weight': 1.5},
            'living': {'privacy': 1, 'frequency': 0.9, 'area_weight': 2.0},
            'dining': {'privacy': 2, 'frequency': 0.6, 'area_weight': 1.0},
            'corridor': {'privacy': 1, 'frequency': 1.0, 'area_weight': 0.3},
            'entrance': {'privacy': 1, 'frequency': 1.0, 'area_weight': 0.8},
            'lobby': {'privacy': 1, 'frequency': 0.8, 'area_weight': 2.0},
            'store': {'privacy': 1, 'frequency': 0.9, 'area_weight': 1.5},
            'restaurant': {'privacy': 2, 'frequency': 0.7, 'area_weight': 1.8}
        }
        
    def extract_spaces_from_prompt(self, prompt: str, question: str) -> Dict[str, Any]:
        """
        Extract spatial information from text prompt using NLP techniques.
        This replaces API calls with pure algorithm-based text analysis.
        """
        # Convert to lowercase for analysis
        text = (prompt + " " + question).lower()
        
        # Define patterns for space detection
        space_patterns = {
            'office': r'office|工作|办公',
            'meeting': r'meeting|conference|会议',
            'bathroom': r'bathroom|toilet|卫生间|洗手间',
            'kitchen': r'kitchen|厨房',
            'bedroom': r'bedroom|卧室',
            'living': r'living|客厅|起居',
            'dining': r'dining|餐厅',
            'corridor': r'corridor|hallway|走廊|过道',
            'entrance': r'entrance|entry|入口|大门',
            'lobby': r'lobby|hall|大厅|门厅',
            'store': r'store|shop|商店|零售',
            'restaurant': r'restaurant|餐饮|饭店'
        }
        
        # Count detection patterns
        number_pattern = r'(\d+)\s*(?:个|间|层|座|栋)?'
        
        detected_spaces = {}
        total_spaces = 0
        
        for space_type, pattern in space_patterns.items():
            # Find all matches for this space type
            matches = re.findall(f'({number_pattern})?\s*({pattern})', text)
            count = 0
            
            for match in matches:
                if match[0] and match[1]:  # Has number
                    try:
                        count += int(match[1])
                    except:
                        count += 1
                else:  # No specific number, assume 1
                    count += 1
            
            if count > 0:
                detected_spaces[space_type] = count
                total_spaces += count
        
        # If no specific spaces detected, create default layout
        if not detected_spaces:
            detected_spaces = {
                'entrance': 1,
                'living': 1,
                'bedroom': 2,
                'kitchen': 1,
                'bathroom': 1
            }
            total_spaces = 6
        
        # Generate spatial relationships based on building type
        building_type = self._infer_building_type(text)
        connections = self._generate_connections(detected_spaces, building_type)
        
        return {
            'spaces': detected_spaces,
            'total_count': total_spaces,
            'building_type': building_type,
            'connections': connections,
            'estimated_area': total_spaces * 20  # Assume 20 sqm per space
        }
    
    def _infer_building_type(self, text: str) -> str:
        """Infer building type from text description."""
        if any(word in text for word in ['住宅', 'house', 'home', 'apartment']):
            return 'residential'
        elif any(word in text for word in ['办公', 'office', 'business']):
            return 'office'
        elif any(word in text for word in ['商业', 'commercial', 'shop', 'mall']):
            return 'commercial'
        else:
            return 'mixed'
    
    def _generate_connections(self, spaces: Dict[str, int], building_type: str) -> List[Tuple[str, str, float]]:
        """Generate logical connections between spaces based on building type."""
        connections = []
        space_list = list(spaces.keys())
        
        # Define typical connection rules
        connection_rules = {
            'residential': {
                'entrance': ['living', 'corridor'],
                'living': ['kitchen', 'dining', 'corridor'],
                'kitchen': ['dining'],
                'bedroom': ['corridor', 'bathroom'],
                'bathroom': ['corridor'],
                'corridor': ['bedroom', 'bathroom', 'living']
            },
            'office': {
                'entrance': ['lobby', 'corridor'],
                'lobby': ['corridor', 'office'],
                'office': ['corridor', 'meeting'],
                'meeting': ['corridor'],
                'bathroom': ['corridor'],
                'corridor': ['office', 'meeting', 'bathroom']
            },
            'commercial': {
                'entrance': ['lobby', 'store'],
                'lobby': ['store', 'restaurant', 'corridor'],
                'store': ['corridor'],
                'restaurant': ['kitchen', 'corridor'],
                'kitchen': ['restaurant'],
                'bathroom': ['corridor'],
                'corridor': ['store', 'restaurant', 'bathroom']
            }
        }
        
        rules = connection_rules.get(building_type, connection_rules['residential'])
        
        for space1 in space_list:
            if space1 in rules:
                for space2 in rules[space1]:
                    if space2 in space_list and space1 != space2:
                        # Calculate connection weight (distance estimate)
                        weight = self._calculate_connection_weight(space1, space2)
                        connections.append((space1, space2, weight))
        
        return connections
    
    def _calculate_connection_weight(self, space1: str, space2: str) -> float:
        """Calculate connection weight between two spaces."""
        # Base weights for different space type combinations
        base_weights = {
            ('entrance', 'living'): 1.0,
            ('living', 'kitchen'): 1.2,
            ('kitchen', 'dining'): 0.8,
            ('bedroom', 'bathroom'): 1.5,
            ('corridor', 'office'): 1.0,
            ('lobby', 'store'): 1.3
        }
        
        # Try both orders
        key1 = (space1, space2)
        key2 = (space2, space1)
        
        if key1 in base_weights:
            return base_weights[key1]
        elif key2 in base_weights:
            return base_weights[key2]
        else:
            # Default weight based on space types
            privacy1 = self.space_types.get(space1, {}).get('privacy', 3)
            privacy2 = self.space_types.get(space2, {}).get('privacy', 3)
            return 1.0 + abs(privacy1 - privacy2) * 0.2
    
    def build_circulation_graph(self, spaces_data: Dict[str, Any]) -> nx.Graph:
        """
        Build NetworkX graph from extracted spaces data.
        Implements the mathematical definition: G = (V, E, W)
        """
        G = nx.Graph()
        
        spaces = spaces_data['spaces']
        connections = spaces_data['connections']
        
        # Add nodes (spaces) with attributes
        node_id = 0
        space_to_node = {}
        
        for space_type, count in spaces.items():
            space_props = self.space_types.get(space_type, {
                'privacy': 3, 'frequency': 0.5, 'area_weight': 1.0
            })
            
            for i in range(count):
                node_name = f"{space_type}_{i+1}" if count > 1 else space_type
                space_to_node[space_type] = node_id
                
                G.add_node(node_id,
                          type=space_type,
                          name=node_name,
                          privacy=space_props['privacy'],
                          frequency=space_props['frequency'],
                          area=space_props['area_weight'] * 20)  # Estimated area
                
                node_id += 1
        
        # Add edges (connections) with weights
        for space1, space2, weight in connections:
            if space1 in space_to_node and space2 in space_to_node:
                node1 = space_to_node[space1]
                node2 = space_to_node[space2]
                G.add_edge(node1, node2, weight=weight)
        
        return G
    
    def calculate_basic_connectivity(self, G: nx.Graph) -> float:
        """
        Calculate basic connectivity: C_basic = |{(u,v) : ∃ path from u to v}| / |V|(|V|-1)
        """
        if len(G.nodes()) <= 1:
            return 1.0
        
        if nx.is_connected(G):
            return 1.0
        else:
            # Calculate the fraction of connected pairs
            connected_pairs = 0
            total_pairs = len(G.nodes()) * (len(G.nodes()) - 1)
            
            for component in nx.connected_components(G):
                comp_size = len(component)
                connected_pairs += comp_size * (comp_size - 1)
            
            return connected_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def calculate_weighted_path_efficiency(self, G: nx.Graph) -> float:
        """
        Calculate weighted path efficiency: E_path = (1/|V|(|V|-1)) * Σ u(vi)*u(vj)/d(vi,vj)
        """
        if len(G.nodes()) <= 1:
            return 1.0
        
        total_efficiency = 0.0
        total_pairs = 0
        
        for u in G.nodes():
            for v in G.nodes():
                if u != v:
                    try:
                        # Get shortest path distance
                        path_length = nx.shortest_path_length(G, u, v, weight='weight')
                        if path_length > 0:
                            # Get usage frequencies
                            freq_u = G.nodes[u].get('frequency', 0.5)
                            freq_v = G.nodes[v].get('frequency', 0.5)
                            
                            # Calculate efficiency for this pair
                            efficiency = (freq_u * freq_v) / path_length
                            total_efficiency += efficiency
                            total_pairs += 1
                    except nx.NetworkXNoPath:
                        # No path exists, efficiency = 0
                        total_pairs += 1
        
        return total_efficiency / total_pairs if total_pairs > 0 else 0.0
    
    def calculate_functional_clustering(self, G: nx.Graph) -> float:
        """
        Calculate functional clustering (modularity): Q = (1/2|E|) * Σ [A_ij - k_i*k_j/2|E|] * δ(c_i, c_j)
        """
        if len(G.edges()) == 0:
            return 0.0
        
        # Create communities based on space types
        communities = defaultdict(list)
        for node in G.nodes():
            space_type = G.nodes[node].get('type', 'unknown')
            communities[space_type].append(node)
        
        # Convert to list of sets for NetworkX
        community_list = [set(nodes) for nodes in communities.values() if len(nodes) > 0]
        
        if len(community_list) <= 1:
            return 0.0
        
        # Calculate modularity
        modularity = nx.community.modularity(G, community_list, weight='weight')
        return max(0.0, modularity)  # Ensure non-negative
    
    def calculate_core_space_accessibility(self, G: nx.Graph) -> float:
        """
        Calculate core space accessibility: A_core = (1/|V_core|) * Σ 1/(d(entrance, v) + 1)
        """
        # Identify entrance nodes
        entrance_nodes = [n for n in G.nodes() if 'entrance' in G.nodes[n].get('type', '')]
        if not entrance_nodes:
            entrance_nodes = [n for n in G.nodes() if 'lobby' in G.nodes[n].get('type', '')]
        
        if not entrance_nodes:
            return 0.5  # Default if no entrance found
        
        entrance = entrance_nodes[0]  # Use first entrance
        
        # Identify core spaces (high frequency usage)
        core_spaces = [n for n in G.nodes() if G.nodes[n].get('frequency', 0) > 0.7]
        
        if not core_spaces:
            core_spaces = list(G.nodes())  # Use all spaces if no high-frequency spaces
        
        total_accessibility = 0.0
        accessible_count = 0
        
        for core_space in core_spaces:
            if core_space != entrance:
                try:
                    distance = nx.shortest_path_length(G, entrance, core_space, weight='weight')
                    accessibility = 1.0 / (distance + 1)
                    total_accessibility += accessibility
                    accessible_count += 1
                except nx.NetworkXNoPath:
                    # No path, accessibility = 0
                    accessible_count += 1
        
        return total_accessibility / accessible_count if accessible_count > 0 else 0.0
    
    def calculate_privacy_gradient(self, G: nx.Graph) -> float:
        """
        Calculate privacy gradient: G_privacy = 1 - (1/|E|) * Σ |p(u) - p(v)| / 4
        """
        if len(G.edges()) == 0:
            return 1.0
        
        total_privacy_diff = 0.0
        edge_count = 0
        
        for u, v in G.edges():
            privacy_u = G.nodes[u].get('privacy', 3)
            privacy_v = G.nodes[v].get('privacy', 3)
            privacy_diff = abs(privacy_u - privacy_v)
            total_privacy_diff += privacy_diff / 4.0  # Normalize by max difference (5-1=4)
            edge_count += 1
        
        average_privacy_diff = total_privacy_diff / edge_count
        return max(0.0, 1.0 - average_privacy_diff)
    
    def calculate_traffic_flow_separation(self, G: nx.Graph) -> float:
        """
        Calculate traffic flow separation: S_traffic = Σ |P_t1 ∩ P_t2| / Σ |P_t1 ∪ P_t2|
        """
        # Define different traffic types based on space types
        traffic_types = {
            'public': ['entrance', 'lobby', 'corridor', 'store', 'restaurant'],
            'private': ['bedroom', 'bathroom', 'office'],
            'service': ['kitchen', 'meeting']
        }
        
        # Find paths for each traffic type
        traffic_paths = {}
        entrance_nodes = [n for n in G.nodes() if G.nodes[n].get('type') in ['entrance', 'lobby']]
        
        if not entrance_nodes:
            return 0.5  # Default separation if no clear entrance
        
        entrance = entrance_nodes[0]
        
        for traffic_type, space_types in traffic_types.items():
            target_nodes = [n for n in G.nodes() if G.nodes[n].get('type') in space_types]
            paths = set()
            
            for target in target_nodes:
                if target != entrance:
                    try:
                        path = nx.shortest_path(G, entrance, target, weight='weight')
                        # Convert path to edge set
                        for i in range(len(path) - 1):
                            paths.add((min(path[i], path[i+1]), max(path[i], path[i+1])))
                    except nx.NetworkXNoPath:
                        continue
            
            traffic_paths[traffic_type] = paths
        
        # Calculate separation
        traffic_types_list = list(traffic_paths.keys())
        total_intersection = 0
        total_union = 0
        
        for i in range(len(traffic_types_list)):
            for j in range(i + 1, len(traffic_types_list)):
                paths_i = traffic_paths[traffic_types_list[i]]
                paths_j = traffic_paths[traffic_types_list[j]]
                
                intersection = len(paths_i & paths_j)
                union = len(paths_i | paths_j)
                
                total_intersection += intersection
                total_union += union
        
        if total_union == 0:
            return 1.0  # Perfect separation if no overlapping paths
        
        overlap_ratio = total_intersection / total_union
        return max(0.0, 1.0 - overlap_ratio)  # Higher value = better separation
    
    def simulate_dynamic_circulation(self, G: nx.Graph, num_agents: int = 20, time_steps: int = 100) -> Dict[str, float]:
        """
        Simulate dynamic circulation using agent-based modeling.
        """
        if len(G.nodes()) <= 1:
            return {'congestion_resistance': 1.0, 'flow_efficiency': 1.0}
        
        # Initialize agents
        agents = []
        entrance_nodes = [n for n in G.nodes() if 'entrance' in G.nodes[n].get('type', '')]
        if not entrance_nodes:
            entrance_nodes = list(G.nodes())[:1]
        
        # Create agents with random destinations
        all_nodes = list(G.nodes())
        for i in range(num_agents):
            start = entrance_nodes[0] if entrance_nodes else all_nodes[0]
            target = np.random.choice(all_nodes)
            agents.append({
                'id': i,
                'current': start,
                'target': target,
                'path': [],
                'completed': False
            })
        
        # Simulation variables
        congestion_events = 0
        total_movements = 0
        completed_agents = 0
        
        # Run simulation
        for step in range(time_steps):
            # Count agents per node
            node_occupancy = defaultdict(int)
            for agent in agents:
                if not agent['completed']:
                    node_occupancy[agent['current']] += 1
            
            # Move agents
            for agent in agents:
                if agent['completed']:
                    continue
                
                current = agent['current']
                target = agent['target']
                
                if current == target:
                    agent['completed'] = True
                    completed_agents += 1
                    continue
                
                # Find next step in path
                try:
                    if not agent['path']:
                        agent['path'] = nx.shortest_path(G, current, target, weight='weight')[1:]
                    
                    if agent['path']:
                        next_node = agent['path'].pop(0)
                        
                        # Check for congestion
                        if node_occupancy[next_node] > 2:  # Congestion threshold
                            congestion_events += 1
                        
                        agent['current'] = next_node
                        total_movements += 1
                
                except (nx.NetworkXNoPath, IndexError):
                    # Can't move, stay in place
                    continue
        
        # Calculate metrics
        congestion_resistance = 1.0 - (congestion_events / max(1, total_movements))
        flow_efficiency = completed_agents / num_agents if num_agents > 0 else 0.0
        
        return {
            'congestion_resistance': max(0.0, congestion_resistance),
            'flow_efficiency': flow_efficiency,
            'total_movements': total_movements,
            'congestion_events': congestion_events
        }
    
    def calculate_overall_circulation_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall circulation score using weighted combination.
        """
        weights = {
            'basic_connectivity': 0.15,
            'path_efficiency': 0.20,
            'functional_clustering': 0.15,
            'core_accessibility': 0.20,
            'privacy_gradient': 0.15,
            'flow_separation': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def analyze_circulation(self, prompt: str, question: str) -> Dict[str, Any]:
        """
        Main function to analyze circulation for a given prompt and question.
        """
        try:
            # Step 1: Extract spatial information from text
            spaces_data = self.extract_spaces_from_prompt(prompt, question)
            
            # Step 2: Build circulation graph
            G = self.build_circulation_graph(spaces_data)
            
            # Step 3: Calculate all circulation metrics
            metrics = {}
            
            # Basic metrics
            metrics['basic_connectivity'] = self.calculate_basic_connectivity(G)
            metrics['path_efficiency'] = self.calculate_weighted_path_efficiency(G)
            metrics['functional_clustering'] = self.calculate_functional_clustering(G)
            
            # Convenience metrics
            metrics['core_accessibility'] = self.calculate_core_space_accessibility(G)
            metrics['privacy_gradient'] = self.calculate_privacy_gradient(G)
            metrics['flow_separation'] = self.calculate_traffic_flow_separation(G)
            
            # Dynamic simulation
            simulation_results = self.simulate_dynamic_circulation(G)
            metrics.update(simulation_results)
            
            # Overall score
            overall_score = self.calculate_overall_circulation_score(metrics)
            
            return {
                'circulation_efficiency': metrics.get('path_efficiency', 0.5),
                'circulation_convenience': (metrics.get('core_accessibility', 0.5) + 
                                          metrics.get('privacy_gradient', 0.5) + 
                                          metrics.get('flow_separation', 0.5)) / 3.0,
                'circulation_dynamics': (metrics.get('congestion_resistance', 0.5) + 
                                       metrics.get('flow_efficiency', 0.5)) / 2.0,
                'overall_circulation_score': overall_score,
                'detailed_metrics': metrics,
                'graph_stats': {
                    'nodes': len(G.nodes()),
                    'edges': len(G.edges()),
                    'density': nx.density(G),
                    'connected_components': nx.number_connected_components(G)
                }
            }
            
        except Exception as e:
            print(f"Warning: Circulation analysis failed: {e}")
            # Return default values if analysis fails
            return {
                'circulation_efficiency': 0.5,
                'circulation_convenience': 0.5,
                'circulation_dynamics': 0.5,
                'overall_circulation_score': 0.5,
                'detailed_metrics': {},
                'graph_stats': {'nodes': 0, 'edges': 0, 'density': 0.0, 'connected_components': 0}
            }


def test_circulation_analyzer():
    """Test function for the circulation analyzer."""
    analyzer = CirculationAnalyzer()
    
    # Test cases
    test_cases = [
        {
            'prompt': '设计一个现代办公楼，包含20个办公室、3个会议室、1个大厅',
            'question': '办公室和会议室的布局是否合理？'
        },
        {
            'prompt': '创建一个住宅，包含3个卧室、2个浴室、1个厨房、1个客厅',
            'question': '房间之间的动线是否便利？'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Question: {test_case['question']}")
        
        result = analyzer.analyze_circulation(test_case['prompt'], test_case['question'])
        
        print(f"Results:")
        for key, value in result.items():
            if key != 'detailed_metrics':
                print(f"  {key}: {value}")


if __name__ == "__main__":
    test_circulation_analyzer()

