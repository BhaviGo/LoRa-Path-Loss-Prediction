import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# ---> THE NEW IMPORT REQUIRED FOR IBM <---
from qiskit_ibm_runtime import QiskitRuntimeService 

# ==========================================
# 1. CONNECT TO THE PHYSICAL QUANTUM COMPUTER
# ==========================================
print("Connecting to IBM Quantum Cloud...")

# ---> PASTE YOUR COPIED API KEY HERE <---
IBM_API_KEY = "P4hKiSMom-r9tZNEH-eP-lQ--FaqKx3KYS0ZbIvaMKOE" 

# Log into IBM's new Runtime system
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_API_KEY)

# Grab that specific empty machine you found
ibm_backend = service.backend('ibm_kingston')

# Connect PennyLane using the modern 'qiskit.remote' command
dev = qml.device(
    'qiskit.remote', 
    wires=4, 
    backend=ibm_backend
)

# ==========================================
# 2. REBUILD THE QUANTUM ARCHITECTURE
# ==========================================
n_qubits = 4
n_layers = 6 

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
# 3. LOAD YOUR PRE-TRAINED "BRAIN"
# ==========================================
print("Loading saved model weights from your laptop...")
model = QuantumRegressor()
model.load_state_dict(torch.load("vqr_lora_model_best.pth"))
model.eval()

# ==========================================
# 4. PREPARE 5 REAL-WORLD DATA ROWS
# ==========================================
print("Extracting 5 diverse test rows from the Prakasam Barrage dataset...")
df = pd.read_csv("Quantum_Ready_LoRa_Data.csv")

X_all = df[['Distance', 'Temp', 'Humidity', 'SF']].values
y_all = df['Path_Loss'].values

test_indices = [10, 200, 450, 700, 950] 

X_sample_tensor = torch.tensor(X_all[test_indices], dtype=torch.float32)
y_actual = y_all[test_indices]

# ==========================================
# 5. EXECUTE ON IBM SUPERCONDUCTING CHIP
# ==========================================
print("\n🚀 Transpiling circuit and sending to IBM Kingston...")
print("Waiting in queue... (Do not close the terminal, this may take a few minutes!)")

with torch.no_grad():
    # Running on the physical hardware!
    ibm_predictions = model(X_sample_tensor).numpy()

# ==========================================
# 6. PRINT RESULTS FOR YOUR PRESENTATION
# ==========================================
print("\n======================================================")
print("🏆 REAL QUANTUM HARDWARE PREDICTION RESULTS 🏆")
print("======================================================")
print(f"{'Row Index':<10} | {'Actual Path Loss':<20} | {'IBM Quantum Prediction':<20}")
print("-" * 54)

for i in range(5):
    row_idx = test_indices[i]
    actual_val = y_actual[i]
    ibm_val = ibm_predictions[i][0]
    
    print(f"Row {row_idx:<6} | {actual_val:<20.4f} | {ibm_val:<20.4f}")

print("======================================================")