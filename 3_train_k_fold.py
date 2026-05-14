import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

# ==========================================
# 1. LOAD THE DATA
# ==========================================
print("Loading Data...")
df = pd.read_csv("Quantum_Ready_LoRa_Data.csv")

# Grab all your features and targets as numpy arrays
X_all = df[['Distance', 'Temp', 'Humidity', 'SF']].values
y_all = df['Path_Loss'].values

# ==========================================
# 2. BUILD THE QUANTUM CIRCUIT (The "Brain")
# ==========================================
n_qubits = 4
n_layers = 6 

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(inputs[:, i], wires=i)
        
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        
        for i in range(n_qubits):
            qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
            
    return qml.expval(qml.PauliZ(0))

class QuantumRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x).view(-1, 1)

# ==========================================
# 3. K-FOLD CROSS VALIDATION SETUP
# ==========================================
k_splits = 5
kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)

epochs = 250 
batch_size = 16

# Trackers for our final grades
fold_metrics = {'R2': [], 'MAE': [], 'RMSE': [], 'MSE': []}
all_folds_loss_history = []
best_r2_score = -float('inf')
best_model_weights = None

print(f"\nStarting {k_splits}-Fold Cross Validation...")
print("Grab a coffee, this will take a few minutes!")

# ==========================================
# 4. THE MASTER TRAINING LOOP
# ==========================================
for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
    print(f"\n--- FOLD {fold + 1}/{k_splits} ---")
    
    # Slice the data for this specific fold
    X_train_tensor = torch.tensor(X_all[train_idx], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_all[train_idx], dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_all[test_idx], dtype=torch.float32)
    y_test_tensor = torch.tensor(y_all[test_idx], dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # CRITICAL: Create a brand new, blank model for each fold
    model = QuantumRegressor()
    loss_function = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    fold_loss_history = []

    # Train the model
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        fold_loss_history.append(avg_loss)
        
        # Only print every 50 epochs to keep your terminal clean
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.4f}")

    all_folds_loss_history.append(fold_loss_history)

    # ==========================================
    # 5. EVALUATE THIS SPECIFIC FOLD
    # ==========================================
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).numpy()
        actual_answers = y_test_tensor.numpy()

    mae = mean_absolute_error(actual_answers, test_predictions)
    mse = mean_squared_error(actual_answers, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_answers, test_predictions)

    print(f"  Fold {fold + 1} Results -> R2: {r2:.4f} | MAE: {mae:.4f}")

    # Save metrics
    fold_metrics['R2'].append(r2)
    fold_metrics['MAE'].append(mae)
    fold_metrics['RMSE'].append(rmse)
    fold_metrics['MSE'].append(mse)

    # If this fold got the highest R2 score, save its brain!
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model_weights = copy.deepcopy(model.state_dict())

# ==========================================
# 6. FINAL THESIS METRICS & SAVING
# ==========================================
print("\n==========================================")
print("🏆 FINAL K-FOLD AVERAGES (Thesis Ready) 🏆")
print("==========================================")
print(f"Average R-Squared (R2): {np.mean(fold_metrics['R2']):.4f}")
print(f"Average MAE:            {np.mean(fold_metrics['MAE']):.4f}")
print(f"Average RMSE:           {np.mean(fold_metrics['RMSE']):.4f}")
print(f"Average MSE:            {np.mean(fold_metrics['MSE']):.4f}")
print("==========================================")

# Save the absolute best model from the 5 folds
print("\nSaving the ultimate best quantum model...")
torch.save(best_model_weights, "vqr_lora_model_best.pth")
print("Success! Model")