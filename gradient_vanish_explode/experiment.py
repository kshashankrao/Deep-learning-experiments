import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Helper Function to Run Experiments (MODIFIED to track ALL layers) ---

def run_experiment(name, activation_fn, learning_rate, num_layers=10, steps=50):
    """
    Runs a training simulation and records the gradient norm of ALL layer weights.
    Returns:
    - first_gradient_norms (list): Norms for the first layer across steps (for time series plot)
    - last_gradient_norms (list): Norms for the last layer across steps (for time series plot)
    - all_layer_norms_data (dict): {layer_index: [norm_step1, norm_step2, ...]}
    """
    print(f"\n--- Running Experiment: {name} ---")
    
    # Define the Deep Model
    layers = []
    input_size = 10
    hidden_size = 50
    
    current_input_size = input_size
    # Create a deep network (10 hidden layers, 9 hidden + 1 output = 10 layers with weights)
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_input_size, hidden_size))
        layers.append(activation_fn)
        current_input_size = hidden_size
    
    # Output layer
    layers.append(nn.Linear(hidden_size, 1))
    
    model = nn.Sequential(*layers)
    
    # Setup Optimizer and Loss
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Data Collection Structures
    first_gradient_norms = []
    last_gradient_norms = []

    all_layer_norms_data = {}
    
    weights_map = {}
    layer_index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights_map[layer_index] = param
            all_layer_norms_data[layer_index] = []
            layer_index += 1
            
    # Get the weight tensors for tracking first and last layer (Layer 0 and Layer 9)
    first_layer_weights = weights_map[0]
    last_layer_weights = weights_map[layer_index - 1]
    
    # Training Loop Simulation
    for step in range(steps):
        X = torch.randn(16, 10)
        Y = torch.randn(16, 1)

        output = model(X)
        loss = criterion(output, Y)
        optimizer.zero_grad()
        loss.backward()

        # Gradient Analysis: Collect ALL Layer Norms 
        for index, weight_tensor in weights_map.items():
            if weight_tensor.grad is not None:
                norm = weight_tensor.grad.data.norm(2).item()
                all_layer_norms_data[index].append(norm)

        # Separate tracking for the first and last layer (for the time-series plot)
        first_gradient_norms.append(all_layer_norms_data[0][-1])
        last_gradient_norms.append(all_layer_norms_data[layer_index - 1][-1])

        optimizer.step()
        
    print(f"Final Loss: {loss.item():.4f}")
    
    return first_gradient_norms, last_gradient_norms, all_layer_norms_data


# Sigmoid Experiment
v_sigmoid_f, v_sigmoid_l, v_sigmoid_all_norms = run_experiment(
    name="Vanishing Gradient (Sigmoid)",
    activation_fn=nn.Sigmoid(), 
    learning_rate=0.01,
    num_layers=10)

# Tanh Experiment
v_tanh_f, v_tanh_l, v_tanh_all_norms = run_experiment(
    name="Vanishing Gradient (Tanh)",
    activation_fn=nn.Tanh(), 
    learning_rate=0.01,
    num_layers=10)

# Exploding Experiment
e_relu_f, e_relu_l, e_relu_all_norms = run_experiment(
    name="Exploding Gradient (ReLU & High LR)",
    activation_fn=nn.ReLU(), 
    learning_rate=100.0, 
    num_layers=10)

# Calculate Average Gradients per Layer 

def calculate_average_norms(all_layer_norms_data):
    avg_norms = {}
    for layer_index, norms_list in all_layer_norms_data.items():
        # Handle cases where list might be empty (shouldn't happen here, but safe practice)
        if norms_list:
            avg_norms[layer_index + 1] = np.mean(norms_list)
        else:
            avg_norms[layer_index + 1] = 0.0

    return avg_norms

avg_sigmoid = calculate_average_norms(v_sigmoid_all_norms)
avg_tanh = calculate_average_norms(v_tanh_all_norms)
avg_exploding = calculate_average_norms(e_relu_all_norms)

# Visualization (Combined Time-Series and Depth Analysis)

# Organize data for time-series plots
time_series_results = [
    ("Vanishing (Sigmoid)", v_sigmoid_f, v_sigmoid_l),
    ("Vanishing (Tanh)", v_tanh_f, v_tanh_l),
    ("Exploding (ReLU)", e_relu_f, e_relu_l)
]

# Organize data for depth plots
depth_analysis_results = [
    ("Vanishing (Sigmoid)", avg_sigmoid),
    ("Vanishing (Tanh)", avg_tanh),
    ("Exploding (ReLU)", avg_exploding)
]

# Create a figure with 3 rows and 2 columns
fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharey='row') # Share Y-axis across rows
fig.suptitle('Comprehensive Gradient Analysis: Time vs. Depth', fontsize=18)
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Plot Column 1: Time-Series (Stability over steps)
for i, (title, first_norms, last_norms) in enumerate(time_series_results):
    ax = axes[i, 0]
    min_len = min(len(first_norms), len(last_norms))
    steps_range = np.arange(min_len)

    ax.plot(steps_range, last_norms[:min_len], 'g-', label='Last Layer Grads (Baseline)', linewidth=2, alpha=0.7)
    ax.plot(steps_range, first_norms[:min_len], 'r--', label='First Layer Grads (Affected)', linewidth=2.5)

    ax.set_title(f'{title} - Stability Over Time', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylabel('L2 Norm (Log Scale)')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Set appropriate Y-limits and thresholds for the time-series plots
    if 'Exploding' in title:
        ax.set_ylim(bottom=1e-8)
        ax.axhline(y=1e2, color='k', linestyle=':', alpha=0.5, label='Exploding Threshold')
    else:
        ax.set_ylim(bottom=1e-10, top=1e1)
        ax.axhline(y=1e-6, color='k', linestyle=':', alpha=0.5, label='Vanishing Threshold')
        
    if i == 2:
        ax.set_xlabel('Training Step')

# Plot Column 2: Depth Analysis (Average Gradient per Layer)
for i, (title, avg_norms) in enumerate(depth_analysis_results):
    ax = axes[i, 1]
    
    layers = list(avg_norms.keys())
    norms = list(avg_norms.values())
    
    ax.plot(layers, norms, 'b-o', label='Avg Gradient Norm', linewidth=2)
    
    ax.set_title(f'{title} - Norm by Layer Depth', fontsize=12)
    ax.set_yscale('log')
    ax.set_xticks(layers)
    ax.set_ylabel('Average L2 Norm (Log Scale)')
    ax.set_xlim(0.5, len(layers) + 0.5) # Set limits to bracket layer numbers
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Add context lines for Vanishing/Exploding
    if 'Exploding' in title:
        ax.axhline(y=1e2, color='r', linestyle='--', alpha=0.6, label='Exploding Threshold')
    else:
        ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.6, label='Vanishing Threshold')
        
    if i == 2:
        ax.set_xlabel('Layer Number (1 = Input, 10 = Output)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('gradient_analysis.png')

