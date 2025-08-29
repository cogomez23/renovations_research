import pandas as pd
import numpy as np

# Read the CSV data
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

# Focus on leased units
leased_df = df_csv[df_csv["leased"] == 1].copy()
leased_df = leased_df.dropna(
    subset=["market_rent_numeric", "bedrooms_numeric", "square_feet_numeric"]
)

print("=" * 80)
print("DISTRICT RENOVATION ANALYSIS - FINAL RECOMMENDATION")
print("=" * 80)

# Calculate current District rents
district_data = leased_df[leased_df["property_name"] == "ICO District"]
district_1br = district_data[district_data["bedrooms_numeric"] == 1][
    "market_rent_numeric"
].mean()
district_2br = district_data[district_data["bedrooms_numeric"] == 2][
    "market_rent_numeric"
].mean()

print(f"\nCURRENT DISTRICT RENTS (Leased Units):")
print(f"1BR Average: ${district_1br:,.0f}")
print(f"2BR Average: ${district_2br:,.0f}")

# Define premium comp targets based on analysis
# Using top 4 comparable premium properties that show strong leasing velocity
premium_comps = [
    "NOVEL Daybreak by Crescent Communities",
    "Parc Ridge",
    "Soleil Lofts",
    "Upper West",
]

print(f"\nPREMIUM COMPARABLE ANALYSIS:")
print(f"Selected Premium Comps: {', '.join(premium_comps)}")

# Calculate average premium comp rents
premium_data = leased_df[leased_df["property_name"].isin(premium_comps)]
premium_1br = premium_data[premium_data["bedrooms_numeric"] == 1][
    "market_rent_numeric"
].mean()
premium_2br = premium_data[premium_data["bedrooms_numeric"] == 2][
    "market_rent_numeric"
].mean()

print(f"\nPREMIUM COMP AVERAGE RENTS:")
print(f"1BR Average: ${premium_1br:,.0f}")
print(f"2BR Average: ${premium_2br:,.0f}")

# Calculate potential rent uplift
uplift_1br = premium_1br - district_1br
uplift_2br = premium_2br - district_2br

print(f"\nPOTENTIAL RENT UPLIFT:")
print(
    f"1BR Uplift: ${uplift_1br:,.0f} ({uplift_1br / district_1br * 100:.1f}% increase)"
)
print(
    f"2BR Uplift: ${uplift_2br:,.0f} ({uplift_2br / district_2br * 100:.1f}% increase)"
)

# Calculate unit mix for District
district_1br_count = len(district_data[district_data["bedrooms_numeric"] == 1])
district_2br_count = len(district_data[district_data["bedrooms_numeric"] == 2])
total_district_units = district_1br_count + district_2br_count

print(f"\nDISTRICT UNIT MIX (from leased data):")
print(
    f"1BR Units: {district_1br_count} ({district_1br_count / total_district_units * 100:.1f}%)"
)
print(
    f"2BR Units: {district_2br_count} ({district_2br_count / total_district_units * 100:.1f}%)"
)
print(f"Total Units: {total_district_units}")

# Calculate weighted average rent uplift
weighted_uplift = (
    uplift_1br * district_1br_count + uplift_2br * district_2br_count
) / total_district_units

print(f"\nWEIGHTED AVERAGE RENT UPLIFT: ${weighted_uplift:.0f}")

# 7% Return Analysis
print(f"\n" + "=" * 60)
print("7% RETURN ON INVESTMENT ANALYSIS")
print("=" * 60)

# Calculate maximum renovation budget for 7% return
max_budget_per_unit = weighted_uplift * 12 / 0.07  # Annual rent increase / 7% return

print(f"\nRENOVATION BUDGET CALCULATION:")
print(f"Annual Rent Increase: ${weighted_uplift * 12:,.0f}")
print(
    f"Required 7% Return: ${weighted_uplift * 12:,.0f} ÷ 0.07 = ${max_budget_per_unit:,.0f}"
)
print(f"Maximum Renovation Budget per Unit: ${max_budget_per_unit:,.0f}")

# Leasing velocity analysis
print(f"\n" + "=" * 60)
print("LEASING VELOCITY & MARKET SHARE ANALYSIS")
print("=" * 60)

district_leases = len(district_data)
premium_leases = len(premium_data)
total_market_leases = district_leases + premium_leases

print(f"\nLEASING VOLUME ANALYSIS:")
print(f"District Leases in Dataset: {district_leases}")
print(f"Premium Comp Leases: {premium_leases}")
print(f"Total Market: {total_market_leases}")
print(f"District Market Share: {district_leases / total_market_leases * 100:.1f}%")

# Assuming District needs 100-150 leases per year (from meeting notes)
annual_lease_requirement = 125  # midpoint
market_multiplier = premium_leases / district_leases

print(f"\nMARKET CAPACITY ANALYSIS:")
print(f"District Annual Lease Requirement: ~{annual_lease_requirement} units")
print(f"Premium Market Leasing Volume: {market_multiplier:.1f}x District's volume")
print(
    f"Estimated Premium Market Annual Capacity: ~{premium_leases * (annual_lease_requirement / district_leases):.0f} leases"
)

# Final recommendation
print(f"\n" + "=" * 80)
print("FINAL RENOVATION RECOMMENDATION")
print("=" * 80)

if max_budget_per_unit >= 6000:  # Reasonable renovation budget threshold
    recommendation = "RECOMMEND RENOVATIONS"
    reasoning = f"""
    ✓ Premium comps support ${weighted_uplift:.0f}/month rent increase
    ✓ Market can absorb renovated units (premium comps lease {market_multiplier:.1f}x District volume)
    ✓ 7% return achievable with ${max_budget_per_unit:,.0f} renovation budget per unit
    ✓ District currently captures only {district_leases / total_market_leases * 100:.1f}% of premium market
    """
else:
    recommendation = "DO NOT RECOMMEND RENOVATIONS"
    reasoning = f"""
    ✗ Renovation budget of ${max_budget_per_unit:,.0f} may be insufficient for meaningful upgrades
    ✗ Rent uplift of ${weighted_uplift:.0f}/month may not justify renovation costs
    ✗ Consider focusing on leasing velocity at current rent levels
    """

print(f"\nRECOMMENDATION: {recommendation}")
print(f"\nREASONING:{reasoning}")

print(f"\nKEY METRICS SUMMARY:")
print(f"• Potential Monthly Rent Increase: ${weighted_uplift:.0f}")
print(f"• Maximum Renovation Budget (7% return): ${max_budget_per_unit:,.0f}")
print(f"• Premium Market Size: {market_multiplier:.1f}x District's current volume")
print(
    f"• District's Current Market Share: {district_leases / total_market_leases * 100:.1f}%"
)
