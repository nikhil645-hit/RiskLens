import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Setup plotting style
sns.set_theme(style="whitegrid")
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/raw/application_train.csv")

print("\n--- 1. Data Shape ---")
print(f"Shape: {df.shape}")

print("\n--- 2. Class Balance (TARGET) ---")
balance = df["TARGET"].value_counts(normalize=True) * 100
print(balance)

# Plot 1: Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="TARGET", palette="viridis")
plt.title("Target Variable Distribution (0 = Repaid, 1 = Default)")
plt.savefig(f"{output_dir}/01_target_distribution.png", bbox_inches='tight')
plt.close()

print("\n--- 3. Top 20 Columns by Missing Values ---")
missing = df.isnull().mean() * 100
top_missing = missing.sort_values(ascending=False).head(20)
print(top_missing)

# Plot 2: Missing Values
plt.figure(figsize=(10, 6))
sns.barplot(x=top_missing.values, y=top_missing.index, palette="mako")
plt.title("Top 20 Columns by Missing Values (%)")
plt.xlabel("Percentage Missing")
plt.savefig(f"{output_dir}/02_missing_values.png", bbox_inches='tight')
plt.close()

print("\n--- 4. Categorical Columns and Unique Counts ---")
cat_cols = df.select_dtypes(include=['object', 'category']).columns
cat_counts = df[cat_cols].nunique().sort_values(ascending=False)
print(cat_counts)

print("\n--- 5. ORGANIZATION_TYPE Default Rate ---")
org_rate = df.groupby("ORGANIZATION_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
print(org_rate.head(10))

# Plot 3: Organization Type Default Rate
plt.figure(figsize=(12, 10))
sns.barplot(x=org_rate.values, y=org_rate.index, palette="rocket")
plt.title("Default Rate by Organization Type (%)")
plt.xlabel("Default Rate (%)")
plt.savefig(f"{output_dir}/03_org_type_default_rate.png", bbox_inches='tight')
plt.close()

print("\n--- 6. The DAYS_EMPLOYED Anomaly ---")
print(f"Max DAYS_EMPLOYED: {df['DAYS_EMPLOYED'].max()}")
print(f"Number of anomalies (365243): {(df['DAYS_EMPLOYED'] == 365243).sum()}")

# Plot 4: Distributions of EXT_SOURCES and DAYS_EMPLOYED
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Anomaly removed for visualization
valid_days = df[df['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED'] / -365
sns.histplot(valid_days, bins=50, ax=axes[0], color="blue", kde=True)
axes[0].set_title("Years Employed (Anomaly Removed)")
axes[0].set_xlabel("Years")

for i, ext in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'], 1):
    sns.histplot(data=df, x=ext, hue="TARGET", bins=50, ax=axes[i], kde=True, common_norm=False, stat="density", palette="viridis")
    axes[i].set_title(f"Distribution of {ext}")

plt.tight_layout()
plt.savefig(f"{output_dir}/04_ext_sources_and_employment.png", bbox_inches='tight')
plt.close()

print("\nEDA Completed successfully. Charts saved to reports/figures/.")
