import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ==========================================
# 1. Environment Setup & Paths
# ==========================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Detected Google Colab environment!")
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive'
else:
    print("Running in local environment.")
    BASE_DIR = '.'

BASELINE_PATH = os.path.join(BASE_DIR, 'baseline.pth')
PRUNED_PATH = os.path.join(BASE_DIR, 'pruned.pth')

# ==========================================
# 2. ResNet Model Architecture Definition
# ==========================================
def get_resnet18_cifar(num_classes=5):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==========================================
# 3. Rigorous Latency Benchmark Function
# ==========================================
def benchmark_model(model, device, batch_size=64, num_runs=500):
    model.eval()
    model.to(device)
    
    # Generate dummy batch
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # 1. Warm-up Phase
    # (Crucial on GPU to allow CUDA graphs / kernels to initialize and cache)
    print(f"Warming up model on {device}...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    # 2. Timing Phase
    print(f"Profiling {num_runs} runs...")
    latencies = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize() # Ensure GPU finished processing before measurement
            end = time.time()
            
            # Record latency in milliseconds
            latencies.append((end - start) * 1000.0)
            
    latencies = np.array(latencies)
    
    # 3. Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    # Throughput = Total Images / Total Time in seconds
    total_time_seconds = np.sum(latencies) / 1000.0
    throughput = (num_runs * batch_size) / total_time_seconds
    
    return {
        'mean': mean_latency,
        'std': std_latency,
        'median': median_latency,
        'p95': p95_latency,
        'throughput': throughput
    }

# ==========================================
# 4. Main Timing Analysis Comparison
# ==========================================
def main():
    # Detect available devices
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
        
    print(f"Timing analysis starting. Available devices to test: {devices}\n")
    
    # Check if files exist
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Baseline model not found at {BASELINE_PATH}")
    if not os.path.exists(PRUNED_PATH):
        raise FileNotFoundError(f"Pruned model not found at {PRUNED_PATH}")
        
    # Load baseline model
    print("Loading Baseline model...")
    baseline_model = get_resnet18_cifar(num_classes=5)
    baseline_model.load_state_dict(torch.load(BASELINE_PATH, map_location='cpu', weights_only=False))
    
    # Load pruned model object
    print("Loading Pruned model (full module serialization format)...")
    pruned_model = torch.load(PRUNED_PATH, map_location='cpu')
    
    batch_size = 64
    num_runs = 500
    
    for device in devices:
        print("\n" + "="*60)
        print(f"TIMING COMPARISON ON: {device.type.upper()}")
        print(f"Batch Size: {batch_size} | Trials: {num_runs}")
        print("="*60)
        
        # Benchmark Baseline
        base_stats = benchmark_model(baseline_model, device, batch_size, num_runs)
        
        # Benchmark Pruned
        pruned_stats = benchmark_model(pruned_model, device, batch_size, num_runs)
        
        # Calculate changes
        latency_diff_pct = ((pruned_stats['mean'] / base_stats['mean']) - 1.0) * 100.0
        throughput_diff_pct = ((pruned_stats['throughput'] / base_stats['throughput']) - 1.0) * 100.0
        
        print("\n" + "-"*60)
        print(f"{'Metric':<25} | {'Baseline':<12} | {'Pruned':<12} | {'Change (%)':<10}")
        print("-"*60)
        print(f"{'Mean Latency (ms)':<25} | {base_stats['mean']:<12.3f} | {pruned_stats['mean']:<12.3f} | {latency_diff_pct:+.1f}%")
        print(f"{'Median Latency (ms)':<25} | {base_stats['median']:<12.3f} | {pruned_stats['median']:<12.3f} | {((pruned_stats['median']/base_stats['median'])-1)*100:+.1f}%")
        print(f"{'P95 Latency (ms)':<25} | {base_stats['p95']:<12.3f} | {pruned_stats['p95']:<12.3f} | {((pruned_stats['p95']/base_stats['p95'])-1)*100:+.1f}%")
        print(f"{'Latency StdDev (ms)':<25} | {base_stats['std']:<12.3f} | {pruned_stats['std']:<12.3f} | -")
        print(f"{'Throughput (img/sec)':<25} | {base_stats['throughput']:<12.1f} | {pruned_stats['throughput']:<12.1f} | {throughput_diff_pct:+.1f}%")
        print("="*60)

if __name__ == '__main__':
    main()
