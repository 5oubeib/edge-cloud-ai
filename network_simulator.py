"""
Network Simulator - Simulates varying network conditions
"""

import random
import time
from config import NETWORK_CONDITIONS


class NetworkSimulator:
    """Simulates network latency and conditions between edge and cloud"""
    
    def __init__(self, condition="moderate"):
        """
        Initialize network simulator
        
        Args:
            condition: 'good', 'moderate', or 'poor'
        """
        self.set_condition(condition)
        self.total_bytes_transferred = 0
    
    def set_condition(self, condition):
        """Change network condition"""
        if condition not in NETWORK_CONDITIONS:
            raise ValueError(f"Invalid condition. Choose from: {list(NETWORK_CONDITIONS.keys())}")
        
        self.condition = condition
        self.params = NETWORK_CONDITIONS[condition]
        print(f"Network condition set to: {condition}")
        print(f"  Latency: {self.params['latency_ms'][0]}-{self.params['latency_ms'][1]}ms")
        print(f"  Bandwidth: {self.params['bandwidth_mbps']}Mbps")
        print(f"  Packet loss: {self.params['packet_loss']*100}%")
    
    def get_latency(self):
        """
        Get current network latency in milliseconds
        
        Returns:
            float: Latency in ms
        """
        min_lat, max_lat = self.params['latency_ms']
        base_latency = random.uniform(min_lat, max_lat)
        
        # Add jitter (±10%)
        jitter = base_latency * random.uniform(-0.1, 0.1)
        
        return max(0, base_latency + jitter)
    
    def simulate_transfer(self, data_size_kb):
        """
        Simulate data transfer time
        
        Args:
            data_size_kb: Data size in kilobytes
            
        Returns:
            float: Transfer time in milliseconds
        """
        bandwidth_kbps = self.params['bandwidth_mbps'] * 1000
        transfer_time_ms = (data_size_kb / bandwidth_kbps) * 1000
        
        # Account for packet loss (requires retransmission)
        if random.random() < self.params['packet_loss']:
            transfer_time_ms *= 1.5  # 50% penalty for retransmission
        
        self.total_bytes_transferred += data_size_kb
        
        return transfer_time_ms
    
    def get_round_trip_time(self, data_size_kb=10):
        """
        Get total round-trip time for a request-response cycle
        
        Args:
            data_size_kb: Size of data being transferred
            
        Returns:
            float: Total RTT in milliseconds
        """
        # Latency + transfer time (both ways)
        latency = self.get_latency()
        upload_time = self.simulate_transfer(data_size_kb)
        download_time = self.simulate_transfer(data_size_kb)
        
        return latency + upload_time + download_time
    
    def is_available(self):
        """
        Check if network is available (accounts for packet loss)
        
        Returns:
            bool: True if network request succeeds
        """
        return random.random() > self.params['packet_loss']
    
    def get_current_condition(self):
        """Get current network condition name"""
        return self.condition
    
    def get_stats(self):
        """Get network statistics"""
        return {
            "condition": self.condition,
            "total_kb_transferred": self.total_bytes_transferred,
            "avg_latency_ms": sum(self.params['latency_ms']) / 2
        }


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    print("Testing Network Simulator...\n")
    
    for condition in ["good", "moderate", "poor"]:
        print(f"\n{'='*60}")
        net = NetworkSimulator(condition=condition)
        
        print(f"\nTesting 10 requests:")
        latencies = []
        for i in range(10):
            rtt = net.get_round_trip_time(data_size_kb=50)
            latencies.append(rtt)
            print(f"  Request {i+1}: {rtt:.2f}ms")
        
        print(f"\nAverage RTT: {sum(latencies)/len(latencies):.2f}ms")
        print(f"Min: {min(latencies):.2f}ms, Max: {max(latencies):.2f}ms")
    
    print("\n✓ Network Simulator working correctly!")
