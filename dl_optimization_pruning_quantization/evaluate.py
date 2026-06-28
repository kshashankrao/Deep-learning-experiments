import os
import time
import torch
import torch.nn as nn
from torchinfo import summary

# Import from modular local codebase
from dataset import get_dataloaders
from model import get_resnet18_cifar
from train import evaluate

# Path configuration
BASELINE_PATH = 'baseline.pth'
PRUNED_PATH = 'pruned.pth'

def measure_latency(model, device, num_runs=100, batch_size=64):
    """
    Measures inference latency in milliseconds per batch.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Latency timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    elapsed = time.time() - start_time
    latency_per_batch_ms = (elapsed / num_runs) * 1000.0
    return latency_per_batch_ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on device: {device}\n")
    
    # Setup data loader and loss criterion
    _, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    
    # Check if models exist
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Baseline weights not found at {BASELINE_PATH}. Please run train.py first.")
    if not os.path.exists(PRUNED_PATH):
        raise FileNotFoundError(f"Pruned model not found at {PRUNED_PATH}. Please run prune.py first.")
    
    # ----------------------------------------
    # Profile Baseline Model
    # ----------------------------------------
    baseline_model = get_resnet18_cifar(num_classes=5).to(device)
    baseline_model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    
    # Reusing evaluate from train.py
    _, base_acc = evaluate(baseline_model, test_loader, criterion, device)
    base_summary = summary(baseline_model, input_size=(1, 3, 32, 32), verbose=0)
    base_params = base_summary.total_params
    base_macs = base_summary.total_mult_adds
    base_latency = measure_latency(baseline_model, device)
    base_size_mb = os.path.getsize(BASELINE_PATH) / (1024 * 1024)
    
    # ----------------------------------------
    # Profile Pruned Model
    # ----------------------------------------
    # Load the physically pruned model object
    pruned_model = torch.load(PRUNED_PATH, map_location=device).to(device)
    
    # Reusing evaluate from train.py
    _, pruned_acc = evaluate(pruned_model, test_loader, criterion, device)
    pruned_summary = summary(pruned_model, input_size=(1, 3, 32, 32), verbose=0)
    pruned_params = pruned_summary.total_params
    pruned_macs = pruned_summary.total_mult_adds
    pruned_latency = measure_latency(pruned_model, device)
    pruned_size_mb = os.path.getsize(PRUNED_PATH) / (1024 * 1024)
    
    # ----------------------------------------
    # Display Comparison Table
    # ----------------------------------------
    print("="*65)
    print(f"{'Metric':<25} | {'Baseline':<15} | {'Pruned':<15} | {'Change (%)':<10}")
    print("="*65)
    print(f"{'Test Accuracy (%)':<25} | {base_acc:<15.2f} | {pruned_acc:<15.2f} | {pruned_acc - base_acc:+.2f}%")
    print(f"{'Parameters (M)':<25} | {base_params/1e6:<15.4f} | {pruned_params/1e6:<15.4f} | {(pruned_params/base_params - 1)*100:+.1f}%")
    print(f"{'FLOPs/MACs (M)':<25} | {base_macs/1e6:<15.4f} | {pruned_macs/1e6:<15.4f} | {(pruned_macs/base_macs - 1)*100:+.1f}%")
    print(f"{'Disk Size (MB)':<25} | {base_size_mb:<15.2f} | {pruned_size_mb:<15.2f} | {(pruned_size_mb/base_size_mb - 1)*100:+.1f}%")
    print(f"{'Batch Latency (ms)':<25} | {base_latency:<15.2f} | {pruned_latency:<15.2f} | {(pruned_latency/base_latency - 1)*100:+.1f}%")
    print("="*65)

if __name__ == '__main__':
    main()
