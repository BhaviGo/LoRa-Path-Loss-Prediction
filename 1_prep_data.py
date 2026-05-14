import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. List of your 5 saved files (Make sure these match your actual file names!)
file_names = [
    'Lora data sf7.csv',      # SF7
    'Lora data sf8.csv', # SF8
    'Lora data sf9.csv', 
    'Lora data sf10.csv', 
    'Lora data sf11.csv'
]

print("Merging files...")
all_data = []
for file in file_names:
    try:
        df = pd.read_csv(file)
        all_data.append(df)
    except FileNotFoundError:
        print(f"Warning: Could not find {file}")

# Combine them all into one giant table
master_df = pd.concat(all_data, ignore_index=True)

# 2. Select the 4 features for our 4 Qubits, plus our Target (Path Loss)
features = ['Distance', 'Temp', 'Humidity', 'SF']
target = ['Path_Loss']

# 3. Scale the features to be between 0 and Pi (3.14) for the Quantum rotations
feature_scaler = MinMaxScaler(feature_range=(0, 3.14159))
master_df[features] = feature_scaler.fit_transform(master_df[features])

# Scale the Path Loss between -1 and 1 (Standard for Quantum Outputs)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
master_df[target] = target_scaler.fit_transform(master_df[target])

# 4. Save the final, Quantum-Ready dataset
master_df.to_csv("Quantum_Ready_LoRa_Data.csv", index=False)
print(f"Success! Merged {len(master_df)} rows and saved as 'Quantum_Ready_LoRa_Data.csv'.")