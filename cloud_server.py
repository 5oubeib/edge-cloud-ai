"""
Cloud Server Simulator - High-accuracy model inference with unlimited resources
"""

import torch
import torchvision.models as models
import time
import random
from config import CLOUD_SERVER, DEVICE, CLOUD_MODEL_ACCURACY


class CloudServer:
    """Simulates a cloud server with high-performance resources"""
    
    def __init__(self):
        """Initialize cloud server"""
        print("Initializing Cloud Server...")
        
        # Load high-accuracy model (ResNet-50)
        print("  Loading ResNet-50 model...")
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.model = self.model.to(DEVICE)
        
        # Statistics
        self.total_inferences = 0
        self.inference_times = []
        self.total_data_received_kb = 0
        
        print("✓ Cloud Server ready")
    
    def inference(self, input_tensor, data_size_kb=50):
        """
        Run inference on cloud server
        
        Args:
            input_tensor: Input tensor (batch_size, 3, 224, 224)
            data_size_kb: Size of input data in KB (for tracking)
            
        Returns:
            dict: {
                'prediction': class_id,
                'confidence': confidence_score,
                'inference_time_ms': time_taken,
                'probabilities': all_class_probabilities
            }
        """
        # Simulate processing time (faster than edge)
        min_time, max_time = CLOUD_SERVER['inference_time_ms']
        simulated_time = random.uniform(min_time, max_time)
        
        # Actual inference
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        actual_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Use simulated time (cloud servers have consistent performance)
        inference_time = simulated_time
        
        # Update statistics
        self.total_inferences += 1
        self.inference_times.append(inference_time)
        self.total_data_received_kb += data_size_kb
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'inference_time_ms': inference_time,
            'energy_consumed_mah': 0,  # Cloud has unlimited power
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def get_stats(self):
        """Get server statistics"""
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        
        return {
            'total_inferences': self.total_inferences,
            'total_data_received_kb': self.total_data_received_kb,
            'avg_inference_time_ms': avg_inference_time,
            'expected_accuracy': CLOUD_MODEL_ACCURACY
        }
    
    def reset(self):
        """Reset server statistics"""
        self.total_inferences = 0
        self.inference_times = []
        self.total_data_received_kb = 0
        print("Cloud server reset to initial state")


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("Testing Cloud Server...\n")
    
    # Create cloud server
    cloud = CloudServer()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    
    print("\nRunning 10 test inferences:")
    for i in range(10):
        result = cloud.inference(dummy_input)
        print(f"  Inference {i+1}:")
        print(f"    Prediction: Class {result['prediction']}")
        print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    Time: {result['inference_time_ms']:.2f}ms")
    
    print("\n" + "="*60)
    stats = cloud.get_stats()
    print("\nCloud Server Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Cloud Server working correctly!")
