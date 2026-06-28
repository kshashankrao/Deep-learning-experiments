import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
from torchinfo import summary

# Import modules from our modular local codebase
from dataset import get_dataloaders
from model import get_resnet18_cifar
from train import train_one_epoch, evaluate

# Path configuration
BASELINE_PATH = 'baseline.pth'
PRUNED_PATH = 'pruned.pth'

def measure_latency(model, device, num_runs=50, batch_size=64):
    """
    Measures inference latency in milliseconds per batch on target hardware.
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Warmup runs to stabilize latency
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
            
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    return ((time.time() - start) / num_runs) * 1000.0

def get_prunable_layers(model):
    """
    Returns named prunable convolutions (excluding downsample skip projection layers).
    """
    prunable_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' not in name and name != 'conv1':
            prunable_layers[name] = module
    return prunable_layers

# ==================================================
# 1. Modular Layer Sensitivity Profiler
# ==================================================
def profile_layer_sensitivity(test_loader, criterion, device, sparsities=[0.2, 0.4]):
    """
    Systematically prunes one layer group at a time to profile sensitivity and speed gains.
    """
    print("\n" + "="*50)
    print("RUNNING LAYER SENSITIVITY & LATENCY PROFILE")
    print("="*50)
    
    example_input = torch.randn(1, 3, 32, 32).to(device)
    importance = tp.importance.MagnitudeImportance(p=1)
    
    # Load baseline
    model = get_resnet18_cifar(num_classes=5).to(device)
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    
    # Reusing train.py's evaluation function
    _, base_acc = evaluate(model, test_loader, criterion, device)
    base_latency = measure_latency(model, device)
    
    print(f"Baseline Accuracy: {base_acc:.2f}% | Latency: {base_latency:.2f} ms/batch\n")
    print(f"{'Layer Name':<25} | {'Sparsity':<10} | {'Test Acc (%)':<15} | {'Acc Drop':<10} | {'Latency Save':<12}")
    print("-"*82)
    
    prunable_layers = get_prunable_layers(model)
    sensitivity_results = []
    
    for layer_name, target_module in prunable_layers.items():
        for sparsity in sparsities:
            # Recreate model & load baseline weights
            temp_model = get_resnet18_cifar(num_classes=5).to(device)
            temp_model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
            
            # Restrict pruning to only this layer
            ignored_layers = []
            for name, m in temp_model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)) and m != getattr(temp_model, layer_name, None):
                    parts = name.split('.')
                    curr = temp_model
                    for p in parts:
                        curr = getattr(curr, p, None) if curr is not None else None
                    if curr == m and name != layer_name:
                        ignored_layers.append(m)
            ignored_layers.append(temp_model.fc)
            
            pruner = tp.pruner.MetaPruner(
                temp_model,
                example_input,
                importance=importance,
                pruning_ratio=sparsity,
                ignored_layers=ignored_layers,
            )
            
            try:
                pruner.step()
                _, acc = evaluate(temp_model, test_loader, criterion, device)
                latency = measure_latency(temp_model, device)
                acc_drop = base_acc - acc
                latency_save = base_latency - latency
                
                print(f"{layer_name:<25} | {sparsity:<10.1f} | {acc:<15.2f} | {acc_drop:<10.2f} | {latency_save:<12.3f} ms")
                
                if sparsity == 0.4:
                    sensitivity_results.append({
                        'layer_name': layer_name,
                        'acc_drop': acc_drop,
                        'latency_save': latency_save
                    })
            except Exception as e:
                print(f"{layer_name:<25} | {sparsity:<10.1f} | Pruning failed: {e}")
                
    sensitivity_results = sorted(sensitivity_results, key=lambda x: x['acc_drop'])
    print("\n" + "="*50)
    print("SENSITIVITY RANKING (Least Sensitive -> Most Sensitive)")
    print("="*50)
    for idx, res in enumerate(sensitivity_results):
        print(f"{idx+1}. {res['layer_name']:<20} | Acc Drop: {res['acc_drop']:.2f}% | Latency Saved: {res['latency_save']:.3f} ms")
    return sensitivity_results

# ==================================================
# 2. Modular Iterative Pruning & Recovery Loop
# ==================================================
def iterative_pruning(train_loader, test_loader, criterion, device, target_ratio=0.5, num_steps=5):
    """
    SOTA practice: Prunes model progressively, fine-tuning for 1 epoch after each step
    to recover intermediate accuracy. Reuses train_one_epoch and evaluate from train.py.
    """
    print("\n" + "="*50)
    print("STARTING ITERATIVE PRUNING & RECOVERY LOOP")
    print("="*50)
    
    model = get_resnet18_cifar(num_classes=5).to(device)
    model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    
    example_input = torch.randn(1, 3, 32, 32).to(device)
    importance = tp.importance.MagnitudeImportance(p=1)
    
    step_ratio = 1.0 - (1.0 - target_ratio) ** (1.0 / num_steps)
    print(f"Target: Prune {target_ratio*100:.1f}% channels in {num_steps} steps.")
    print(f"Pruning step size: {step_ratio*100:.2f}% per iteration.\n")
    
    for step in range(1, num_steps + 1):
        # Initialize MetaPruner
        ignored_layers = [model.fc]
        pruner = tp.pruner.MetaPruner(
            model,
            example_input,
            importance=importance,
            pruning_ratio=step_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.step()
        
        # Profile current structure
        sum_info = summary(model, input_size=(1, 3, 32, 32), verbose=0)
        params = sum_info.total_params
        macs = sum_info.total_mult_adds
        
        # Reusing evaluate from train.py
        _, acc_before = evaluate(model, test_loader, criterion, device)
        
        # Reusing train_one_epoch from train.py
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        _, acc_after = evaluate(model, test_loader, criterion, device)
        print(f"Step {step}/{num_steps} | Params: {params/1e6:.3f}M | MACs: {macs/1e6:.3f}M | "
              f"Acc Before: {acc_before:.2f}% -> Acc After Fine-tune: {acc_after:.2f}%")
              
    torch.save(model, PRUNED_PATH)
    print(f"\nSaved final pruned model object to: {PRUNED_PATH}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Run Sensitivity profiling
    profile_layer_sensitivity(test_loader, criterion, device, sparsities=[0.2, 0.4])
    
    # 2. Run Iterative Pruning
    iterative_pruning(train_loader, test_loader, criterion, device, target_ratio=0.5, num_steps=5)

if __name__ == '__main__':
    main()
