import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR

# ==========================================
# 1. LOAD AND SPLIT THE DATA
# ==========================================
print("Loading Data...")
df = pd.read_csv("Quantum_Ready_LoRa_Data.csv")

# Grab all your features and targets
X_all = df[['Distance', 'Temp', 'Humidity', 'SF']].values
y_all = df['Path_Loss'].values

# The Magic Trick: Hide 20% of the data for the Final Exam!
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30, random_state=42)

# Convert only the TRAINING data to PyTorch tensors for the DataLoader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Create a data loader to feed the model in batches (Using only the 80% Training Data)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ==========================================
# 2. BUILD THE QUANTUM CIRCUIT (The "Brain")
# ==========================================
n_qubits = 4
n_layers = 6 # Bumped to 6 for slightly more expressivity!

# Create a simulated quantum device
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # 'inputs' are your 4 weather/distance metrics
    # 'weights' are the tunable knobs the model learns to adjust
    
    for layer in range(n_layers):
        # A. Encode Data (Rotate qubits by the data angles)
        for i in range(n_qubits):
            qml.RX(inputs[:, i], wires=i)
        
        # B. Entangle Qubits (Link weather and distance together!)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        
        # C. Apply Trainable Weights
        for i in range(n_qubits):
            qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            
    # D. Measure the output (Returns a value between -1 and 1)
    return qml.expval(qml.PauliZ(0))

# ==========================================
# 3. CREATE THE PYTORCH MODEL
# ==========================================
class QuantumRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Create random starting weights (n_layers, n_qubits, 3 rotations)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x).view(-1, 1)

model = QuantumRegressor()
loss_function = nn.MSELoss() # Standard error measurement
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
# Cuts the learning rate by 50% every 50 epochs
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
epochs = 250 # Number of times to loop through the whole dataset
print("\nStarting Quantum Training...")

loss_history = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        
        # Make a prediction
        predictions = model(batch_X)
        
        # Check how wrong the prediction was
        loss = loss_function(predictions, batch_y)
        
        # Adjust the quantum weights to be smarter next time
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    # Take a step with the scheduler at the end of the epoch!
    scheduler.step()
        
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs} | Error (Loss): {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.4f}")

# ==========================================
# 5. EVALUATE ON THE HIDDEN TEST DATA
# ==========================================
print("\nEvaluating Final Model on Hidden Test Data (The Final Exam)...")

# Convert the hidden 20% testing data into tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    # Force the model to guess the Path Loss for data it has NEVER seen
    test_predictions = model(X_test_tensor).numpy()

# Calculate the metrics against the real, hidden answers
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_predictions)

print("\n=== FINAL TEST METRICS (Real-World Accuracy) ===")
print(f"Mean Squared Error (MSE):      {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE):     {mae:.4f}")
print(f"R-Squared Score (R2):          {r2:.4f}")


# ==========================================
# 6. GRAPH THE RESULTS
# ==========================================
print("\nGenerating Fading Graph...")

plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='purple', linewidth=2.5)
plt.title("Quantum Model Learning Curve (Path Loss Prediction)")
plt.xlabel("Training Epochs")
plt.ylabel("Prediction Error (MSE)")
plt.grid(True)

# ADD THESE TWO LINES TO SAVE THE GRAPH:
# dpi=300 makes it high-definition for your presentation!
plt.savefig("quantum_learning_curve.png", dpi=300, bbox_inches='tight')
print("Graph saved as 'quantum_learning_curve.png'")

plt.show()
# ==========================================
# 7. SAVE THE TRAINED MODEL
# ==========================================
print("\nSaving the trained quantum model...")

# Save only the learned weights (the "state dictionary")
torch.save(model.state_dict(), "vqr_lora_model.pth")

print("Success! Model saved to 'vqr_lora_model.pth'")