import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read and clean data (same as before)
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

# Fit regression model
X = leased_df_no_novel[["square_feet_numeric", "bedrooms_numeric"]]
y = leased_df_no_novel["market_rent_numeric"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Calculate residuals
leased_df_no_novel["predicted_rent"] = y_pred
leased_df_no_novel["residual"] = (
    leased_df_no_novel["market_rent_numeric"] - leased_df_no_novel["predicted_rent"]
)

# Get property averages for plotting
property_avg = (
    leased_df_no_novel.groupby("property_name")
    .agg(
        {
            "market_rent_numeric": "mean",
            "predicted_rent": "mean",
            "residual": "mean",
            "square_feet_numeric": "mean",
            "bedrooms_numeric": "mean",
        }
    )
    .reset_index()
)

# Create the visualization
plt.figure(figsize=(14, 10))

# Create subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Rental Market Regression Analysis (Excluding NOVEL Daybreak)', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted Rent Scatter
ax1.scatter(property_avg["predicted_rent"], property_avg["market_rent_numeric"], 
           alpha=0.7, s=80, c='steelblue', edgecolors='black', linewidth=0.5)

# Add perfect prediction line (45-degree line)
min_rent = min(property_avg["predicted_rent"].min(), property_avg["market_rent_numeric"].min())
max_rent = max(property_avg["predicted_rent"].max(), property_avg["market_rent_numeric"].max())
ax1.plot([min_rent, max_rent], [min_rent, max_rent], 'r--', linewidth=2, label='Perfect Prediction')

# Highlight District
district_data = property_avg[property_avg["property_name"] == "ICO District"]
if not district_data.empty:
    ax1.scatter(district_data["predicted_rent"], district_data["market_rent_numeric"], 
               s=150, c='red', marker='D', edgecolors='black', linewidth=2, 
               label='ICO District', zorder=5)

ax1.set_xlabel('Predicted Rent ($)', fontweight='bold')
ax1.set_ylabel('Actual Rent ($)', fontweight='bold')
ax1.set_title(f'Actual vs Predicted Rent (R² = {r2:.3f})', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add property labels for ALL properties
for _, row in property_avg.iterrows():
    # Shorten property names for readability
    short_name = row["property_name"].replace(' Apartments', '').replace('ICO ', '').replace(' at Daybreak', '').replace(' by Crescent Communities', '')
    if len(short_name) > 15:
        short_name = short_name[:15] + '...'
    
    ax1.annotate(short_name, 
                (row["predicted_rent"], row["market_rent_numeric"]),
                xytext=(3, 3), textcoords='offset points', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'),
                ha='left')

# 2. Residuals vs Predicted Rent
colors = ['red' if residual > 0 else 'blue' for residual in property_avg["residual"]]
ax2.scatter(property_avg["predicted_rent"], property_avg["residual"], 
           alpha=0.7, s=80, c=colors, edgecolors='black', linewidth=0.5)

# Add zero line
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='$50 Premium')
ax2.axhline(y=-50, color='orange', linestyle='--', alpha=0.5, label='$50 Below Market')

# Highlight District
if not district_data.empty:
    ax2.scatter(district_data["predicted_rent"], district_data["residual"], 
               s=150, c='darkred', marker='D', edgecolors='black', linewidth=2, 
               label='ICO District', zorder=5)

ax2.set_xlabel('Predicted Rent ($)', fontweight='bold')
ax2.set_ylabel('Residual (Actual - Predicted) ($)', fontweight='bold')
ax2.set_title('Premium/Discount vs Market Expectation', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Add property labels to residuals chart
for _, row in property_avg.iterrows():
    short_name = row["property_name"].replace(' Apartments', '').replace('ICO ', '').replace(' at Daybreak', '').replace(' by Crescent Communities', '')
    if len(short_name) > 15:
        short_name = short_name[:15] + '...'
    
    ax2.annotate(short_name, 
                (row["predicted_rent"], row["residual"]),
                xytext=(3, 3), textcoords='offset points', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'),
                ha='left')

# 3. Property Ranking Bar Chart
property_sorted = property_avg.sort_values("residual", ascending=True)
colors_bar = ['red' if x > 0 else 'blue' for x in property_sorted["residual"]]

bars = ax3.barh(range(len(property_sorted)), property_sorted["residual"], color=colors_bar, alpha=0.7)
ax3.set_yticks(range(len(property_sorted)))
ax3.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in property_sorted["property_name"]], 
                   fontsize=8)
ax3.set_xlabel('Residual ($)', fontweight='bold')
ax3.set_title('Properties Ranked by Premium/Discount', fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.grid(True, alpha=0.3, axis='x')

# Highlight District bar
district_idx = property_sorted.index[property_sorted["property_name"] == "ICO District"].tolist()
if district_idx:
    bars[district_idx[0]].set_color('darkred')
    bars[district_idx[0]].set_edgecolor('black')
    bars[district_idx[0]].set_linewidth(2)

# 4. Square Feet vs Rent with Regression Line
# Create a range of square footage for the regression line
sqft_range = np.linspace(leased_df_no_novel["square_feet_numeric"].min(), 
                        leased_df_no_novel["square_feet_numeric"].max(), 100)

# For visualization, assume average bedrooms (2 bedrooms)
avg_bedrooms = leased_df_no_novel["bedrooms_numeric"].mean()
X_line = np.column_stack([sqft_range, np.full(100, avg_bedrooms)])
y_line = model.predict(X_line)

# Plot all individual units
ax4.scatter(leased_df_no_novel["square_feet_numeric"], leased_df_no_novel["market_rent_numeric"], 
           alpha=0.3, s=20, c='lightblue', edgecolors='none')

# Plot property averages
ax4.scatter(property_avg["square_feet_numeric"], property_avg["market_rent_numeric"], 
           alpha=0.8, s=80, c='steelblue', edgecolors='black', linewidth=0.5)

# Plot regression line
ax4.plot(sqft_range, y_line, 'r-', linewidth=2, label=f'Regression Line (avg bedrooms)')

# Highlight District
if not district_data.empty:
    ax4.scatter(district_data["square_feet_numeric"], district_data["market_rent_numeric"], 
               s=150, c='red', marker='D', edgecolors='black', linewidth=2, 
               label='ICO District', zorder=5)

ax4.set_xlabel('Square Feet', fontweight='bold')
ax4.set_ylabel('Rent ($)', fontweight='bold')
ax4.set_title('Rent vs Square Footage', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add property labels to square footage chart
for _, row in property_avg.iterrows():
    short_name = row["property_name"].replace(' Apartments', '').replace('ICO ', '').replace(' at Daybreak', '').replace(' by Crescent Communities', '')
    if len(short_name) > 15:
        short_name = short_name[:15] + '...'
    
    ax4.annotate(short_name, 
                (row["square_feet_numeric"], row["market_rent_numeric"]),
                xytext=(3, 3), textcoords='offset points', fontsize=7, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'),
                ha='left')

plt.tight_layout()
plt.savefig('rental_regression_analysis_labeled.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'rental_regression_analysis_labeled.png'")
# plt.show()  # Commented out to avoid interactive window

# Print summary statistics
print("="*80)
print("REGRESSION ANALYSIS SUMMARY")
print("="*80)
print(f"Model R²: {r2:.3f}")
print(f"Intercept: ${model.intercept_:.2f}")
print(f"Square Feet Coefficient: ${model.coef_[0]:.2f} per sq ft")
print(f"Bedrooms Coefficient: ${model.coef_[1]:.2f} per bedroom")
print()

print("TOP 5 PREMIUM PROPERTIES (Above Regression Line):")
top_premium = property_avg.nlargest(5, "residual")
for _, row in top_premium.iterrows():
    print(f"  {row['property_name'][:35]:<35} +${row['residual']:>6.0f}")

print("\nTOP 5 BELOW MARKET PROPERTIES:")
bottom_properties = property_avg.nsmallest(5, "residual")
for _, row in bottom_properties.iterrows():
    print(f"  {row['property_name'][:35]:<35} ${row['residual']:>7.0f}")

if not district_data.empty:
    print(f"\nICO DISTRICT POSITION:")
    print(f"  Residual: ${district_data['residual'].iloc[0]:+.0f}")
    print(f"  Actual Rent: ${district_data['market_rent_numeric'].iloc[0]:,.0f}")
    print(f"  Predicted Rent: ${district_data['predicted_rent'].iloc[0]:,.0f}")
    print(f"  Status: {'Premium' if district_data['residual'].iloc[0] > 0 else 'Below Market'}")
