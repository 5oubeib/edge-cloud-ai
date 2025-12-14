"""
Edge Device Simulator - Lightweight model inference with resource constraints
"""

import torch
import torchvision.models as models
import time
import random
from config import EDGE_DEVICE, DEVICE, EDGE_MODEL_ACCURACY


class EdgeDevice:
    """Simulates an edge device with limited resources"""
    
    def __init__(self, battery_level=100.0):
        """
        Initialize edge device
        
        Args:
            battery_level: Initial battery percentage (0-100)
        """
        print("Initializing Edge Device...")
        
        # Battery state
        self.battery_capacity = EDGE_DEVICE['battery_capacity_mah']
        self.battery_level = battery_level  # Percentage
        self.battery_mah = (battery_level / 100.0) * self.battery_capacity
        
        # Load lightweight model (MobileNetV2)
        print("  Loading MobileNetV2 model...")
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        self.model = self.model.to(DEVICE)
        
        # Statistics
        self.total_inferences = 0
        self.total_energy_consumed = 0
        self.inference_times = []
        
        print(f"✓ Edge Device ready (Battery: {battery_level}%)")
    
    def get_battery_level(self):
        """Get current battery level as percentage"""
        return (self.battery_mah / self.battery_capacity) * 100
    
    def consume_energy(self, amount_mah):
        """
        Consume battery energy
        
        Args:
            amount_mah: Energy to consume in mAh
        """
        self.battery_mah = max(0, self.battery_mah - amount_mah)
        self.total_energy_consumed += amount_mah
    
    def can_inference(self):
        """Check if device has enough battery for inference"""
        required_energy = EDGE_DEVICE['inference_cost_mah']
        return self.battery_mah >= required_energy
    
    def inference(self, input_tensor):
        """
        Run inference on edge device
        
        Args:
            input_tensor: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            dict: {
                'prediction': class_id,
                'confidence': confidence_score,
                'inference_time_ms': time_taken,
                'energy_consumed_mah': energy_used,
                'probabilities': all_class_probabilities
            }
        """
        if not self.can_inference():
            raise RuntimeError("Insufficient battery for inference")
        
        # Simulate processing time
        min_time, max_time = EDGE_DEVICE['inference_time_ms']
        simulated_time = random.uniform(min_time, max_time)
        
        # Actual inference
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        actual_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Use simulated time (more realistic for edge devices)
        inference_time = simulated_time
        
        # Consume energy
        energy_consumed = EDGE_DEVICE['inference_cost_mah']
        self.consume_energy(energy_consumed)
        
        # Update statistics
        self.total_inferences += 1
        self.inference_times.append(inference_time)
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'inference_time_ms': inference_time,
            'energy_consumed_mah': energy_consumed,
            'battery_level': self.get_battery_level(),
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def idle(self, duration_minutes):
        """
        Simulate idle power consumption
        
        Args:
            duration_minutes: Idle duration in minutes
        """
        idle_cost = EDGE_DEVICE['idle_cost_per_min_mah'] * duration_minutes
        self.consume_energy(idle_cost)
    
    def charge(self, amount_percent=100):
        """
        Charge battery
        
        Args:
            amount_percent: Amount to charge (percentage)
        """
        self.battery_mah = min(
            self.battery_capacity, 
            self.battery_mah + (amount_percent / 100.0) * self.battery_capacity
        )
        print(f"Battery charged to {self.get_battery_level():.1f}%")
    
    def get_stats(self):
        """Get device statistics"""
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        
        return {
            'total_inferences': self.total_inferences,
            'total_energy_consumed_mah': self.total_energy_consumed,
            'battery_level_percent': self.get_battery_level(),
            'avg_inference_time_ms': avg_inference_time,
            'expected_accuracy': EDGE_MODEL_ACCURACY
        }
    
    def reset(self):
        """Reset device to initial state"""
        self.battery_level = 100.0
        self.battery_mah = self.battery_capacity
        self.total_inferences = 0
        self.total_energy_consumed = 0
        self.inference_times = []
        print("Edge device reset to initial state")


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("Testing Edge Device...\n")
    
    # Create edge device
    edge = EdgeDevice(battery_level=100)
    
    # Create dummy input (CIFAR-10 is 32x32, but models expect 224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    print("\nRunning 10 test inferences:")
    for i in range(10):
        result = edge.inference(dummy_input)
        print(f"  Inference {i+1}:")
        print(f"    Prediction: Class {result['prediction']}")
        print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    Time: {result['inference_time_ms']:.2f}ms")
        print(f"    Energy: {result['energy_consumed_mah']:.4f}mAh")
        print(f"    Battery: {result['battery_level']:.2f}%")
    
    print("\n" + "="*60)
    stats = edge.get_stats()
    print("\nEdge Device Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Edge Device working correctly!")
