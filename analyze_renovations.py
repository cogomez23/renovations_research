import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV data
df_csv = pd.read_csv("../DIS_market_export.csv")
print("CSV Data Shape:", df_csv.shape)
print("Columns:", df_csv.columns.tolist())
print("Unique Properties:", df_csv["property_name"].unique())
print("\nLeased units (leased=1):", df_csv[df_csv["leased"] == 1].shape[0])
print("Total units:", len(df_csv))

# Clean market_rent column and convert to numeric
df_csv["market_rent_clean"] = (
    df_csv["market_rent"].str.replace("$", "").str.replace(",", "").str.strip()
)
df_csv["market_rent_numeric"] = pd.to_numeric(
    df_csv["market_rent_clean"], errors="coerce"
)

# Focus on leased units only (where leased=1)
leased_df = df_csv[df_csv["leased"] == 1].copy()
print("\nLeased Units Analysis:")
print("Leased units shape:", leased_df.shape)
print("Properties with leased units:")
print(leased_df["property_name"].value_counts().head(10))
