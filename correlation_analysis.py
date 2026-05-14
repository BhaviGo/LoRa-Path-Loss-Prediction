import pandas as pd

# Load your real dataset
print("Loading data...")
df = pd.read_csv("Quantum_Ready_LoRa_Data.csv")

# Make sure we are only looking at the specific features for the paper
columns_to_check = ['Path_Loss', 'Distance', 'SF', 'Humidity', 'Temp', 'Wind']

existing_columns = [col for col in columns_to_check if col in df.columns]
df_subset = df[existing_columns]

# Calculate the Pearson Correlation matrix using ONLY pandas (no seaborn/scipy freezing)
correlation_matrix = df_subset.corr(method='pearson')

# Extract just the correlations relating to 'Path_Loss'
path_loss_corr = correlation_matrix['Path_Loss'].sort_values(ascending=False)

print("\n=== REAL PEARSON CORRELATIONS WITH PATH LOSS ===")
print(path_loss_corr)