import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read and clean data
df_csv = pd.read_csv("../DIS_market_export.csv", low_memory=False)

# Clean data
df_csv["market_rent_clean"] = (
    df_csv["market_rent"].str.replace("$", "").str.replace(",", "").str.strip()
)
df_csv["market_rent_numeric"] = pd.to_numeric(
    df_csv["market_rent_clean"], errors="coerce"
)
df_csv["square_feet_clean"] = (
    df_csv["square_feet"].astype(str).str.replace(",", "").str.strip()
)
df_csv["square_feet_numeric"] = pd.to_numeric(
    df_csv["square_feet_clean"], errors="coerce"
)
df_csv["bedrooms_numeric"] = pd.to_numeric(df_csv["bedrooms"], errors="coerce")

# Focus on leased units and clean data
leased_df = df_csv[df_csv["leased"] == 1].copy()
leased_df = leased_df.dropna(
    subset=["market_rent_numeric", "bedrooms_numeric", "square_feet_numeric"]
)

# Exclude NOVEL Daybreak as requested
leased_df_no_novel = leased_df[
    leased_df["property_name"] != "NOVEL Daybreak by Crescent Communities"
].copy()

print("=" * 80)
print("REGRESSION-BASED PREMIUM PROPERTY ANALYSIS")
print("=" * 80)

print(f"\nData Summary:")
print(f"Total leased units: {len(leased_df)}")
print(f"Leased units excluding NOVEL: {len(leased_df_no_novel)}")
print(f"NOVEL leased units: {len(leased_df) - len(leased_df_no_novel)}")

# Prepare features for regression
# Using square feet and bedrooms as predictors of rent
X = leased_df_no_novel[["square_feet_numeric", "bedrooms_numeric"]]
y = leased_df_no_novel["market_rent_numeric"]

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"\nLinear Regression Results (excluding NOVEL):")
print(f"R-squared: {r2:.3f}")
print(f"Intercept: ${model.intercept_:.2f}")
print(f"Coefficient - Square Feet: ${model.coef_[0]:.2f} per sq ft")
print(f"Coefficient - Bedrooms: ${model.coef_[1]:.2f} per bedroom")

# Calculate residuals (actual - predicted)
leased_df_no_novel["predicted_rent"] = y_pred
leased_df_no_novel["residual"] = (
    leased_df_no_novel["market_rent_numeric"] - leased_df_no_novel["predicted_rent"]
)
leased_df_no_novel["residual_pct"] = (
    leased_df_no_novel["residual"] / leased_df_no_novel["predicted_rent"]
) * 100

# Analyze by property
property_analysis = (
    leased_df_no_novel.groupby("property_name")
    .agg(
        {
            "residual": ["mean", "median", "std"],
            "residual_pct": ["mean", "median"],
            "market_rent_numeric": ["mean", "count"],
            "predicted_rent": "mean",
        }
    )
    .round(2)
)

property_analysis.columns = [
    "avg_residual",
    "median_residual",
    "std_residual",
    "avg_residual_pct",
    "median_residual_pct",
    "avg_actual_rent",
    "lease_count",
    "avg_predicted_rent",
]
property_analysis = property_analysis.reset_index()

# Sort by average residual (most premium first)
property_analysis_sorted = property_analysis.sort_values(
    "avg_residual", ascending=False
)

print(f"\n" + "=" * 80)
print("PROPERTIES RANKED BY PREMIUM TO REGRESSION LINE")
print("=" * 80)
print("(Positive residuals = above market line = premium properties)")
print()

for _, row in property_analysis_sorted.iterrows():
    status = "PREMIUM" if row["avg_residual"] > 0 else "BELOW MARKET"
    print(
        f"{row['property_name'][:35]:<35} | "
        f"Residual: ${row['avg_residual']:>6.0f} | "
        f"Pct: {row['avg_residual_pct']:>5.1f}% | "
        f"Leases: {row['lease_count']:>3.0f} | {status}"
    )

# Identify premium properties (significantly above regression line)
# Using properties with positive residuals and sufficient volume
premium_threshold = 0  # Above regression line
min_leases = 20  # Minimum lease volume for reliability

premium_properties = property_analysis_sorted[
    (property_analysis_sorted["avg_residual"] > premium_threshold)
    & (property_analysis_sorted["lease_count"] >= min_leases)
]

print(f"\n" + "=" * 60)
print("IDENTIFIED PREMIUM PROPERTIES (Above Regression Line)")
print("=" * 60)
print(f"Criteria: Residual > ${premium_threshold}, Min {min_leases} leases")
print()

total_premium_leases = 0
for _, row in premium_properties.iterrows():
    total_premium_leases += row["lease_count"]
    print(f"✓ {row['property_name']}")
    print(
        f"  Average Premium: ${row['avg_residual']:,.0f} ({row['avg_residual_pct']:.1f}%)"
    )
    print(
        f"  Actual Rent: ${row['avg_actual_rent']:,.0f} vs Predicted: ${row['avg_predicted_rent']:,.0f}"
    )
    print(f"  Lease Volume: {row['lease_count']:.0f}")
    print()

# Analyze District's position
district_analysis = property_analysis_sorted[
    property_analysis_sorted["property_name"] == "ICO District"
]

if not district_analysis.empty:
    district_row = district_analysis.iloc[0]
    print(f"ICO DISTRICT ANALYSIS:")
    print(
        f"Current Position: ${district_row['avg_residual']:,.0f} residual ({district_row['avg_residual_pct']:.1f}%)"
    )
    print(f"Actual Rent: ${district_row['avg_actual_rent']:,.0f}")
    print(f"Predicted Rent: ${district_row['avg_predicted_rent']:,.0f}")
    print(f"Lease Volume: {district_row['lease_count']:.0f}")

    if district_row["avg_residual"] < 0:
        print("Status: BELOW MARKET - Renovation opportunity confirmed")
    else:
        print("Status: AT/ABOVE MARKET - Limited renovation upside")

print(f"\n" + "=" * 60)
print("RENOVATION POTENTIAL ANALYSIS")
print("=" * 60)

if not premium_properties.empty and not district_analysis.empty:
    # Calculate potential uplift to premium level
    avg_premium_residual = premium_properties["avg_residual"].mean()
    district_residual = district_row["avg_residual"]
    potential_uplift = avg_premium_residual - district_residual

    print(f"Average Premium Property Residual: ${avg_premium_residual:.0f}")
    print(f"District Current Residual: ${district_residual:.0f}")
    print(f"Potential Monthly Rent Uplift: ${potential_uplift:.0f}")
    print(f"Premium Market Lease Volume: {total_premium_leases:.0f}")
    print(f"District Lease Volume: {district_row['lease_count']:.0f}")
    print(
        f"Market Capacity Ratio: {total_premium_leases / district_row['lease_count']:.1f}x"
    )

# Create visualization data for plotting
print(f"\n" + "=" * 60)
print("REGRESSION VISUALIZATION READY")
print("=" * 60)
print("Data prepared for scatter plot showing:")
print("- All properties relative to regression line")
print("- Premium properties (above line) highlighted")
print("- District's current position")
print("- Renovation potential gap")
