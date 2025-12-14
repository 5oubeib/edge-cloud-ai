"""
Configuration file for Edge-Cloud AI Placement Framework
All system parameters and constants in one place
"""

import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Edge Model (Lightweight)
EDGE_MODEL_NAME = "mobilenet_v2"
EDGE_MODEL_ACCURACY = 0.87  # Expected accuracy on CIFAR-10

# Cloud Model (High Accuracy)
CLOUD_MODEL_NAME = "resnet50"
CLOUD_MODEL_ACCURACY = 0.94  # Expected accuracy on CIFAR-10

# ============================================================================
# NETWORK SIMULATION PARAMETERS
# ============================================================================
NETWORK_CONDITIONS = {
    "good": {
        "latency_ms": (20, 50),      # (min, max) in milliseconds
        "bandwidth_mbps": 10,
        "packet_loss": 0.0
    },
    "moderate": {
        "latency_ms": (100, 200),
        "bandwidth_mbps": 2,
        "packet_loss": 0.02
    },
    "poor": {
        "latency_ms": (300, 500),
        "bandwidth_mbps": 0.5,
        "packet_loss": 0.05
    }
}

# ============================================================================
# EDGE DEVICE PARAMETERS
# ============================================================================
EDGE_DEVICE = {
    "battery_capacity_mah": 5000,
    "inference_cost_mah": 0.5,      # Battery cost per inference
    "idle_cost_per_min_mah": 0.1,   # Idle battery drain
    "inference_time_ms": (200, 400), # Processing time range
    "max_concurrent": 1              # Sequential processing
}

# ============================================================================
# CLOUD SERVER PARAMETERS
# ============================================================================
CLOUD_SERVER = {
    "inference_time_ms": (50, 100),  # Processing time range
    "max_concurrent": 100,           # Can handle many requests
    "unlimited_power": True
}

# ============================================================================
# PLACEMENT DECISION THRESHOLDS
# ============================================================================
# A²PF (Accuracy-Aware Placement Framework) Parameters
PLACEMENT_PARAMS = {
    # Confidence threshold for hybrid mode
    "confidence_threshold": 0.80,    # If edge confidence < 80%, offload to cloud
    
    # Network latency threshold (ms)
    "latency_threshold": 250,        # Above this, prefer edge
    
    # Battery level thresholds
    "battery_critical": 20,          # Below 20%: minimize edge usage
    "battery_low": 50,               # Below 50%: selective offloading
    
    # Accuracy requirements
    "accuracy_low": 0.80,            # Low accuracy requirement
    "accuracy_medium": 0.90,         # Medium accuracy requirement
    "accuracy_high": 0.95            # High accuracy requirement
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT = {
    # Dataset
    "dataset": "CIFAR10",
    "num_classes": 10,
    "batch_size": 1,                 # Single inference at a time
    
    # Test scenarios
    "num_requests_per_scenario": 200,  # Reduced for quick testing
    "num_runs": 3,                     # Repeat experiments for statistical validity
    
    # Random seed for reproducibility
    "random_seed": 42
}

# ============================================================================
# BASELINE STRATEGIES
# ============================================================================
BASELINES = {
    "always_edge": "All inference on edge device (Simple baseline)",
    "always_cloud": "All inference on cloud server (Simple baseline)",
    "latency_based": "Choose based on network latency only (Simple heuristic)",
    "gnh": "Greedy Nominator Heuristic (From literature - reliability-focused)",
    "adaptive_resource": "Adaptive Resource-Aware Distribution (From literature)",
    "random": "Random placement (Statistical baseline)",
    "a2pf": "Our proposed Accuracy-Aware Placement Framework (Novel approach)"
}

# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================
PATHS = {
    "data": "./data",
    "models": "./models",
    "results": "./results",
    "logs": "./logs"
}

LOG_EVERY_N_REQUESTS = 50  # Print progress every N requests

# ============================================================================
# METRICS TO TRACK
# ============================================================================
METRICS = [
    "request_id",
    "timestamp",
    "strategy",
    "placement_decision",  # edge/cloud/hybrid
    "network_condition",   # good/moderate/poor
    "network_latency_ms",
    "battery_level",
    "accuracy_requirement",
    "edge_inference_time_ms",
    "cloud_inference_time_ms",
    "total_latency_ms",
    "energy_consumed_mah",
    "prediction_correct",
    "confidence_score",
    "true_label",
    "predicted_label"
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_SETTINGS = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid"
}
