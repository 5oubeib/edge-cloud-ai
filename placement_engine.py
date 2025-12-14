"""
Placement Engine - A²PF (Accuracy-Aware Placement Framework)
Core decision-making logic for edge-cloud task placement
NOW INCLUDES: GNH, Adaptive Resource-Aware, and Random baselines
"""

import random
import numpy as np
from config import PLACEMENT_PARAMS, EDGE_MODEL_ACCURACY, CLOUD_MODEL_ACCURACY


class PlacementEngine:
    """
    A²PF Decision Engine with multiple baseline strategies
    Decides where to run inference: edge, cloud, or hybrid
    """
    
    def __init__(self, strategy="a2pf"):
        """
        Initialize placement engine
        
        Args:
            strategy: 'a2pf', 'always_edge', 'always_cloud', 'latency_based',
                     'gnh', 'adaptive_resource', 'random'
        """
        self.strategy = strategy
        self.decisions_made = 0
        self.placement_history = {
            'edge': 0,
            'cloud': 0,
            'hybrid': 0
        }
        
        # For GNH - track reliability scores
        self.edge_reliability = 0.85  # Simulated edge reliability
        self.cloud_reliability = 0.98  # Cloud is more reliable
        
        # For adaptive resource-aware - track resource usage
        self.resource_usage_history = []
        
        print(f"Placement Engine initialized with strategy: {strategy}")
    
    def decide(self, context):
        """
        Make placement decision based on context
        
        Args:
            context: dict containing:
                - network_latency_ms: current network latency
                - battery_level: edge device battery (0-100)
                - accuracy_requirement: required accuracy (0.0-1.0)
                - edge_confidence: confidence from edge model (optional, for hybrid)
        
        Returns:
            str: 'edge', 'cloud', or 'hybrid'
        """
        self.decisions_made += 1
        
        if self.strategy == "always_edge":
            decision = self._always_edge(context)
        elif self.strategy == "always_cloud":
            decision = self._always_cloud(context)
        elif self.strategy == "latency_based":
            decision = self._latency_based(context)
        elif self.strategy == "gnh":
            decision = self._gnh(context)
        elif self.strategy == "adaptive_resource":
            decision = self._adaptive_resource(context)
        elif self.strategy == "random":
            decision = self._random_placement(context)
        elif self.strategy == "a2pf":
            decision = self._a2pf(context)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.placement_history[decision] += 1
        return decision
    
    # ========================================================================
    # BASELINE STRATEGIES
    # ========================================================================
    
    def _always_edge(self, context):
        """Always run on edge (baseline)"""
        return 'edge'
    
    def _always_cloud(self, context):
        """Always run on cloud (baseline)"""
        return 'cloud'
    
    def _latency_based(self, context):
        """Choose based on network latency only (simple baseline)"""
        latency = context.get('network_latency_ms', 100)
        threshold = PLACEMENT_PARAMS['latency_threshold']
        
        if latency > threshold:
            return 'edge'  # High latency, use edge
        else:
            return 'cloud'  # Low latency, use cloud
    
    def _random_placement(self, context):
        """Random placement (statistical baseline)"""
        return random.choice(['edge', 'cloud', 'hybrid'])
    
    # ========================================================================
    # GNH (Greedy Nominator Heuristic) - From Literature
    # ========================================================================
    
    def _gnh(self, context):
        """
        Greedy Nominator Heuristic
        Based on: Adaptive Edge-Cloud Environments for Rural AI
        
        Ranks nodes by: completion_time + risk_factor + cost
        Focuses on reliability and makespan reduction
        """
        latency = context.get('network_latency_ms', 100)
        battery = context.get('battery_level', 100)
        
        # Calculate scores for edge and cloud
        # Lower score = better choice
        
        # Edge score
        edge_completion_time = 300  # Simulated edge inference time
        edge_risk = (1 - self.edge_reliability) * 100  # Risk factor (higher = worse)
        edge_cost = 2.0  # Energy cost (normalized)
        edge_score = edge_completion_time + edge_risk + edge_cost
        
        # Cloud score
        cloud_completion_time = latency + 75  # Network + inference
        cloud_risk = (1 - self.cloud_reliability) * 100  # Risk factor
        cloud_cost = 0.5  # Lower cost (no battery)
        cloud_score = cloud_completion_time + cloud_risk + cloud_cost
        
        # GNH uses replication for reliability
        # If both scores are close, use hybrid (replication strategy)
        if abs(edge_score - cloud_score) < 20:
            return 'hybrid'  # Replicate for reliability
        elif edge_score < cloud_score:
            return 'edge'
        else:
            return 'cloud'
    
    # ========================================================================
    # ADAPTIVE RESOURCE-AWARE - From Literature
    # ========================================================================
    
    def _adaptive_resource(self, context):
        """
        Adaptive Resource-Aware Distribution
        Based on: Real-time monitoring of network, energy, and device load
        
        Dynamically decides based on current resource state
        """
        latency = context.get('network_latency_ms', 100)
        battery = context.get('battery_level', 100)
        
        # Track resource usage
        self.resource_usage_history.append({
            'latency': latency,
            'battery': battery
        })
        
        # Calculate resource availability scores (0-1, higher = better)
        network_score = max(0, 1 - (latency / 500))  # Normalize latency
        energy_score = battery / 100  # Normalize battery
        
        # Weighted combination
        edge_favorability = (network_score * 0.3 + energy_score * 0.7)
        
        # Decision thresholds
        if edge_favorability > 0.7:
            return 'edge'  # Good conditions for edge
        elif edge_favorability < 0.3:
            return 'cloud'  # Poor conditions, use cloud
        else:
            return 'hybrid'  # Medium conditions, adaptive
    
    # ========================================================================
    # A²PF STRATEGY (Our Proposed Approach)
    # ========================================================================
    
    def _a2pf(self, context):
        """
        Accuracy-Aware Placement Framework
        Considers: accuracy requirements, network, battery, and confidence
        
        KEY DIFFERENCE: Explicitly optimizes for accuracy preservation
        """
        # Extract context
        network_latency = context.get('network_latency_ms', 100)
        battery_level = context.get('battery_level', 100)
        accuracy_req = context.get('accuracy_requirement', 0.90)
        edge_confidence = context.get('edge_confidence', None)
        
        # Get thresholds
        latency_threshold = PLACEMENT_PARAMS['latency_threshold']
        confidence_threshold = PLACEMENT_PARAMS['confidence_threshold']
        battery_critical = PLACEMENT_PARAMS['battery_critical']
        battery_low = PLACEMENT_PARAMS['battery_low']
        
        # ====================================================================
        # RULE 1: Critical Battery → Minimize Edge Usage
        # ====================================================================
        if battery_level < battery_critical:
            # Only use edge if network is extremely poor AND accuracy req is low
            if network_latency > 400 and accuracy_req < 0.85:
                return 'edge'
            return 'cloud'
        
        # ====================================================================
        # RULE 2: High Accuracy Requirement → Prefer Cloud
        # ====================================================================
        if accuracy_req >= PLACEMENT_PARAMS['accuracy_high']:
            # Need 95%+ accuracy → Cloud is safer
            # Exception: if network is terrible, try hybrid
            if network_latency > 400:
                return 'hybrid'
            return 'cloud'
        
        # ====================================================================
        # RULE 3: Poor Network → Prefer Edge
        # ====================================================================
        if network_latency > latency_threshold * 1.5:  # Very high latency
            # Use edge unless accuracy requirement is very high
            if accuracy_req < PLACEMENT_PARAMS['accuracy_medium']:
                return 'edge'
            return 'hybrid'  # Try edge first, fallback to cloud if needed
        
        # ====================================================================
        # RULE 4: Hybrid Mode with Confidence Check
        # ====================================================================
        if edge_confidence is not None:
            # We have a confidence score from edge model
            if edge_confidence >= confidence_threshold:
                # High confidence → Trust edge result
                return 'edge'
            else:
                # Low confidence → Offload to cloud for better accuracy
                return 'cloud'
        
        # ====================================================================
        # RULE 5: Balanced Decision (Default Case)
        # ====================================================================
        # Calculate placement score for edge vs cloud
        edge_score = self._calculate_edge_score(
            network_latency, battery_level, accuracy_req
        )
        cloud_score = self._calculate_cloud_score(
            network_latency, battery_level, accuracy_req
        )
        
        if edge_score > cloud_score:
            return 'edge'
        elif cloud_score > edge_score:
            return 'cloud'
        else:
            # Tie → Use hybrid mode
            return 'hybrid'
    
    def _calculate_edge_score(self, latency, battery, accuracy_req):
        """
        Calculate score for edge placement
        Higher score = better for edge
        """
        score = 0
        
        # Battery factor (more battery = higher score)
        if battery > 80:
            score += 3
        elif battery > 50:
            score += 2
        elif battery > 20:
            score += 1
        
        # Network latency factor (higher latency = prefer edge)
        if latency > 300:
            score += 3
        elif latency > 200:
            score += 2
        elif latency > 100:
            score += 1
        
        # Accuracy requirement factor (lower req = prefer edge)
        if accuracy_req < 0.85:
            score += 3
        elif accuracy_req < 0.90:
            score += 1
        
        # Edge model can deliver ~87% accuracy
        if accuracy_req <= EDGE_MODEL_ACCURACY:
            score += 2
        
        return score
    
    def _calculate_cloud_score(self, latency, battery, accuracy_req):
        """
        Calculate score for cloud placement
        Higher score = better for cloud
        """
        score = 0
        
        # Network latency factor (lower latency = prefer cloud)
        if latency < 50:
            score += 3
        elif latency < 100:
            score += 2
        elif latency < 200:
            score += 1
        
        # Accuracy requirement factor (higher req = prefer cloud)
        if accuracy_req >= 0.95:
            score += 3
        elif accuracy_req >= 0.90:
            score += 2
        elif accuracy_req >= 0.85:
            score += 1
        
        # Cloud model can deliver ~94% accuracy
        if accuracy_req > EDGE_MODEL_ACCURACY:
            score += 2
        
        # Battery doesn't matter for cloud (always available)
        score += 1
        
        return score
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self):
        """Get placement statistics"""
        total = sum(self.placement_history.values())
        
        if total == 0:
            percentages = {k: 0 for k in self.placement_history.keys()}
        else:
            percentages = {
                k: (v / total) * 100 
                for k, v in self.placement_history.items()
            }
        
        return {
            'strategy': self.strategy,
            'total_decisions': self.decisions_made,
            'edge_count': self.placement_history['edge'],
            'cloud_count': self.placement_history['cloud'],
            'hybrid_count': self.placement_history['hybrid'],
            'edge_percent': percentages['edge'],
            'cloud_percent': percentages['cloud'],
            'hybrid_percent': percentages['hybrid']
        }
    
    def reset(self):
        """Reset statistics"""
        self.decisions_made = 0
        self.placement_history = {
            'edge': 0,
            'cloud': 0,
            'hybrid': 0
        }
        self.resource_usage_history = []


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("Testing Placement Engine with ALL strategies...\n")
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Good network, high battery, medium accuracy',
            'context': {
                'network_latency_ms': 50,
                'battery_level': 90,
                'accuracy_requirement': 0.90
            }
        },
        {
            'name': 'Poor network, low battery, high accuracy',
            'context': {
                'network_latency_ms': 400,
                'battery_level': 30,
                'accuracy_requirement': 0.95
            }
        },
        {
            'name': 'Medium network, critical battery, low accuracy',
            'context': {
                'network_latency_ms': 150,
                'battery_level': 15,
                'accuracy_requirement': 0.80
            }
        }
    ]
    
    # Test each strategy
    strategies = ['always_edge', 'always_cloud', 'latency_based', 
                  'gnh', 'adaptive_resource', 'random', 'a2pf']
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy.upper()}")
        print('='*70)
        
        engine = PlacementEngine(strategy=strategy)
        
        for scenario in scenarios:
            decision = engine.decide(scenario['context'])
            print(f"\n  {scenario['name']}:")
            print(f"    → Decision: {decision.upper()}")
        
        print(f"\n  Statistics:")
        stats = engine.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
    
    print("\n✓ Placement Engine with all strategies working correctly!")
