import pandas as pd
import numpy as np
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
print("REVISED DISTRICT RENOVATION ANALYSIS")
print("REGRESSION-BASED PREMIUM PROPERTY IDENTIFICATION")
print("=" * 80)

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

# Get property averages
property_analysis = (
    leased_df_no_novel.groupby("property_name")
    .agg(
        {
            "residual": ["mean", "std"],
            "market_rent_numeric": ["mean", "count"],
            "predicted_rent": "mean",
            "bedrooms_numeric": "mean",
        }
    )
    .round(2)
)

property_analysis.columns = [
    "avg_residual",
    "std_residual",
    "avg_actual_rent",
    "lease_count",
    "avg_predicted_rent",
    "avg_bedrooms",
]
property_analysis = property_analysis.reset_index()

# Identify regression-based premium properties
# Criteria: Positive residual, minimum lease volume for statistical significance
min_leases = 50  # Increased threshold for reliability
premium_properties_df = property_analysis[
    (property_analysis["avg_residual"] > 0)
    & (property_analysis["lease_count"] >= min_leases)
].sort_values("avg_residual", ascending=False)

print(f"\nREGRESSION MODEL PERFORMANCE:")
print(f"R-squared: {r2:.3f}")
print(f"Model explains {r2 * 100:.1f}% of rent variation")
print(
    f"Rent = ${model.intercept_:.0f} + ${model.coef_[0]:.2f}*sqft + ${model.coef_[1]:.0f}*bedrooms"
)

print(f"\nIDENTIFIED PREMIUM PROPERTIES (Above Regression Line):")
print(
    f"Criteria: Positive residual, minimum {min_leases} leases for statistical reliability"
)
print("-" * 80)

premium_property_names = []
total_premium_leases = 0

for _, row in premium_properties_df.iterrows():
    premium_property_names.append(row["property_name"])
    total_premium_leases += row["lease_count"]
    print(f"✓ {row['property_name']}")
    print(f"  Premium over market expectation: ${row['avg_residual']:,.0f}")
    print(
        f"  Actual rent: ${row['avg_actual_rent']:,.0f} vs Predicted: ${row['avg_predicted_rent']:,.0f}"
    )
    print(f"  Statistical reliability: {row['lease_count']:.0f} leases")
    print()

# Analyze District's position
district_analysis = property_analysis[
    property_analysis["property_name"] == "ICO District"
]

if not district_analysis.empty:
    district_row = district_analysis.iloc[0]
    print(f"ICO DISTRICT CURRENT POSITION:")
    print(
        f"Residual: ${district_row['avg_residual']:+.0f} (already {'+' if district_row['avg_residual'] > 0 else ''}premium)"
    )
    print(f"Actual rent: ${district_row['avg_actual_rent']:,.0f}")
    print(f"Predicted rent: ${district_row['avg_predicted_rent']:,.0f}")
    print(f"Lease sample size: {district_row['lease_count']:.0f}")

    # Calculate realistic renovation potential
    if not premium_properties_df.empty:
        avg_premium_residual = premium_properties_df["avg_residual"].mean()
        district_residual = district_row["avg_residual"]
        realistic_uplift = avg_premium_residual - district_residual

        print(f"\nREALISTIC RENOVATION POTENTIAL:")
        print(f"Average premium property residual: ${avg_premium_residual:.0f}")
        print(f"District current residual: ${district_residual:.0f}")
        print(f"Potential monthly rent increase: ${realistic_uplift:.0f}")

        # 7% ROI Analysis
        annual_increase = realistic_uplift * 12
        max_renovation_budget = annual_increase / 0.07

        print(f"\n7% ROI ANALYSIS:")
        print(f"Annual rent increase: ${annual_increase:,.0f}")
        print(f"Maximum renovation budget: ${max_renovation_budget:,.0f}")
        print(
            f"Budget assessment: {'SUFFICIENT' if max_renovation_budget >= 10000 else 'INSUFFICIENT'} for meaningful renovations"
        )

        # Market context (with caveats)
        print(f"\nMARKET CONTEXT (Dataset-based, see limitations below):")
        print(f"Premium properties in dataset: {len(premium_properties_df)} properties")
        print(f"Premium property lease volume in dataset: {total_premium_leases}")
        print(f"District lease volume in dataset: {district_row['lease_count']:.0f}")
        print(
            f"Premium-to-District lease ratio: {total_premium_leases / district_row['lease_count']:.1f}x"
        )

print(f"\n" + "=" * 80)
print("CRITICAL DATA LIMITATIONS & DISCLAIMERS")
print("=" * 80)

print("""
⚠️  MISSING CRITICAL DATA:
• No vacancy rates or time-to-lease data
• No lease term information (short vs long-term leases)
• No temporal data (when leases occurred, seasonal effects)
• No concession details (effective vs gross rents)
• No property age, condition, or recent renovation status
• No amenity details or location quality factors

⚠️  ANALYSIS LIMITATIONS:
• "Market absorption" = lease count ratios in dataset, NOT true market capacity
• "Leasing velocity" = NOT calculated (would require time-series data)
• Dataset represents snapshot, not market dynamics over time
• High lease counts could indicate high turnover (negative) or high demand (positive)
• Regression model only accounts for bedrooms + square footage (missing key variables)

⚠️  RENOVATION RECOMMENDATION CAVEATS:
• Premium properties may have amenities District cannot replicate
• Renovation costs vary significantly by scope and property condition  
• Market conditions may have changed since data collection
• Regulatory, zoning, or physical constraints not considered
• Competition response to District renovations not modeled

⚠️  REQUIRED ADDITIONAL ANALYSIS:
• Property condition assessments and renovation feasibility studies
• Current market vacancy rates and absorption trends
• Detailed amenity gap analysis between District and premium comps
• Financial modeling with multiple scenarios and sensitivity analysis
• Legal and regulatory review of renovation possibilities
""")

# Final recommendation framework
if not district_analysis.empty and not premium_properties_df.empty:
    print(f"\n" + "=" * 80)
    print("PRELIMINARY RECOMMENDATION FRAMEWORK")
    print("=" * 80)

    if realistic_uplift > 50 and max_renovation_budget >= 15000:
        recommendation = "PROCEED WITH DETAILED FEASIBILITY STUDY"
        reasoning = f"""
✓ Regression analysis suggests ${realistic_uplift:.0f}/month potential uplift
✓ 7% ROI achievable with ${max_renovation_budget:,.0f} budget
✓ Multiple premium comparables validate higher rent levels
→ NEXT STEP: Detailed feasibility study addressing data limitations above
        """
    elif realistic_uplift > 25:
        recommendation = "CAUTIOUS PROCEED - LIMITED UPSIDE"
        reasoning = f"""
⚠ Limited rent uplift potential (${realistic_uplift:.0f}/month)
⚠ Renovation budget of ${max_renovation_budget:,.0f} may not justify improvements
→ NEXT STEP: Focus on operational improvements vs capital renovations
        """
    else:
        recommendation = "DO NOT RENOVATE - FOCUS ON OPERATIONS"
        reasoning = f"""
✗ District already performing at/above market expectations
✗ Limited renovation upside (${realistic_uplift:.0f}/month)
→ NEXT STEP: Optimize operations, marketing, and tenant retention
        """

    print(f"\nPRELIMINARY RECOMMENDATION: {recommendation}")
    print(f"{reasoning}")

    print(f"\nKEY METRICS SUMMARY:")
    print(
        f"• District current position: ${district_row['avg_residual']:+.0f} vs market expectation"
    )
    print(f"• Realistic monthly rent increase potential: ${realistic_uplift:.0f}")
    print(f"• Maximum renovation budget (7% ROI): ${max_renovation_budget:,.0f}")
    print(f"• Premium comparables identified: {len(premium_properties_df)}")
    print(
        f"• Statistical reliability: Based on {district_row['lease_count']:.0f} District leases"
    )
