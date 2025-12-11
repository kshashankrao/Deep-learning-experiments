import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Setup the Complex Data (Non-Linear) ---
# Non-linear relationship: y = sin(x/5) + noise
X = torch.randn(500, 1) * 15
y = torch.sin(X / 5) + (X / 10) + torch.randn(500, 1) * 0.5
print(f"Data created with {X.shape[0]} points and non-linear relationship.")

# --- 2. Define a Non-Linear Model ---
# A simple neural network with one hidden layer and ReLU activation
class NonLinearModel(nn.Module):
    def __init__(self):
        super(NonLinearModel, self).__init__()
        self.layer_1 = nn.Linear(1, 64) # Input to Hidden Layer
        self.relu = nn.ReLU()           # Non-linear activation
        self.layer_2 = nn.Linear(64, 1) # Hidden Layer to Output

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x

# --- 3. Training Function (Mostly Unchanged) ---
def train_model(X, y, learning_rate, steps=500):
    model = NonLinearModel()
    criterion = nn.MSELoss()
    # Using Adam optimizer, which is more robust for complex problems
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    for step in range(steps):
        y_predicted = model(X)
        loss = criterion(y_predicted, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return loss_history

# --- 4. Define Learning Rates for Dramatic Contrast ---

# Too High (Will likely diverge or oscillate wildly)
HIGH_LR = 0.1

# Optimal (A sensible rate for Adam optimizer)
OPTIMAL_LR = 0.001

# Too Low (Will clearly stagnate and not make much progress)
LOW_LR = 0.00001

print(f"Starting training with three distinct learning rates for 500 steps...")

# Train for each scenario
history_high = train_model(X, y, HIGH_LR, steps=500)
history_optimal = train_model(X, y, OPTIMAL_LR, steps=500)
history_low = train_model(X, y, LOW_LR, steps=500)

# --- 5. Visualization of Loss History ---
plt.figure(figsize=(10, 6))

# Plot the results
plt.plot(history_high, label=f'Too High LR ({HIGH_LR}) - Oscillation/Divergence', color='red', linestyle='--')
plt.plot(history_optimal, label=f'Optimal LR ({OPTIMAL_LR}) - Fast Convergence', color='green', linewidth=2)
plt.plot(history_low, label=f'Too Low LR ({LOW_LR}) - Slow Stagnation', color='blue', linestyle=':')

plt.title('Effect of Learning Rate on Training Loss (Non-Linear Model)')
plt.xlabel('Training Step')
plt.ylabel('Mean Squared Error (MSE) Loss')
plt.ylim(0, max(np.max(history_optimal) * 5, np.max(history_high) * 0.5)) # Adjust y-axis for better visibility
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()