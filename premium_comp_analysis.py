import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV data
df_csv = pd.read_csv('../DIS_market_export.csv', low_memory=False)

# Clean market_rent column and convert to numeric
df_csv['market_rent_clean'] = df_csv['market_rent'].str.replace('$', '').str.replace(',', '').str.strip()
df_csv['market_rent_numeric'] = pd.to_numeric(df_csv['market_rent_clean'], errors='coerce')

# Clean square_feet column and convert to numeric
df_csv['square_feet_clean'] = df_csv['square_feet'].astype(str).str.replace(',', '').str.strip()
df_csv['square_feet_numeric'] = pd.to_numeric(df_csv['square_feet_clean'], errors='coerce')

# Clean bedrooms column
df_csv['bedrooms_numeric'] = pd.to_numeric(df_csv['bedrooms'], errors='coerce')

# Focus on leased units only (where leased=1) - these represent actual executed rents
leased_df = df_csv[df_csv['leased'] == 1].copy()

# Remove rows with missing critical data
leased_df = leased_df.dropna(subset=['market_rent_numeric', 'bedrooms_numeric', 'square_feet_numeric'])

print("=== PHASE 1: PREMIUM COMPS ANALYSIS ===")
print("\n1. Rent Analysis by Property (Leased Units Only)")
print("-" * 60)

# Calculate average rent by property and bedroom count for leased units
rent_by_property = leased_df.groupby(['property_name', 'bedrooms_numeric']).agg({
    'market_rent_numeric': ['mean', 'median', 'count'],
    'square_feet_numeric': 'mean'
}).round(2)

rent_by_property.columns = ['avg_rent', 'median_rent', 'lease_count', 'avg_sqft']
rent_by_property = rent_by_property.reset_index()

# Calculate rent per square foot
rent_by_property['rent_per_sqft'] = (rent_by_property['avg_rent'] / rent_by_property['avg_sqft']).round(2)

# Focus on 1BR and 2BR units (most common)
br1_data = rent_by_property[rent_by_property['bedrooms_numeric'] == 1].copy()
br2_data = rent_by_property[rent_by_property['bedrooms_numeric'] == 2].copy()

print("1-BEDROOM UNITS (Leased Rents):")
br1_sorted = br1_data.sort_values('avg_rent', ascending=False)
print(br1_sorted[['property_name', 'avg_rent', 'median_rent', 'lease_count', 'rent_per_sqft']])

print("\n2-BEDROOM UNITS (Leased Rents):")
br2_sorted = br2_data.sort_values('avg_rent', ascending=False)
print(br2_sorted[['property_name', 'avg_rent', 'median_rent', 'lease_count', 'rent_per_sqft']])

# Identify ICO District's position
district_1br = br1_data[br1_data['property_name'] == 'ICO District']
district_2br = br2_data[br2_data['property_name'] == 'ICO District']

if not district_1br.empty:
    district_1br_rent = district_1br['avg_rent'].iloc[0]
    print(f"\nICO District 1BR Average Leased Rent: ${district_1br_rent:,.0f}")
    
if not district_2br.empty:
    district_2br_rent = district_2br['avg_rent'].iloc[0]
    print(f"ICO District 2BR Average Leased Rent: ${district_2br_rent:,.0f}")

print("\n" + "="*60)
print("PHASE 1: PREMIUM COMP IDENTIFICATION")
print("="*60)

# Define premium comps based on rent levels above District
# Looking for properties that are clearly premium but comparable in market/location

# 1BR Analysis - Premium Comps
print("\n1-BEDROOM PREMIUM COMP ANALYSIS:")
print("-" * 40)

if not district_1br.empty:
    district_1br_rent = district_1br['avg_rent'].iloc[0]
    
    # Identify properties significantly above District (>$50 premium)
    premium_1br = br1_sorted[br1_sorted['avg_rent'] > district_1br_rent + 50]
    print(f"District 1BR Rent: ${district_1br_rent:,.0f}")
    print(f"Properties with >$50 premium over District:")
    
    for _, row in premium_1br.iterrows():
        premium = row['avg_rent'] - district_1br_rent
        print(f"  {row['property_name']}: ${row['avg_rent']:,.0f} (+${premium:,.0f}) - {row['lease_count']} leases")

# 2BR Analysis - Premium Comps  
print("\n2-BEDROOM PREMIUM COMP ANALYSIS:")
print("-" * 40)

if not district_2br.empty:
    district_2br_rent = district_2br['avg_rent'].iloc[0]
    
    # Identify properties significantly above District (>$50 premium)
    premium_2br = br2_sorted[br2_sorted['avg_rent'] > district_2br_rent + 50]
    print(f"District 2BR Rent: ${district_2br_rent:,.0f}")
    print(f"Properties with >$50 premium over District:")
    
    for _, row in premium_2br.iterrows():
        premium = row['avg_rent'] - district_2br_rent
        print(f"  {row['property_name']}: ${row['avg_rent']:,.0f} (+${premium:,.0f}) - {row['lease_count']} leases")

print("\n" + "="*60)
print("LEASING VOLUME ANALYSIS")
print("="*60)

# Calculate District's leasing requirements (mentioned as 100-150 leases per year)
district_total_leases = len(leased_df[leased_df['property_name'] == 'ICO District'])
print(f"\nICO District Total Leased Units in Dataset: {district_total_leases}")

# Analyze premium comp leasing volumes
print(f"\nPremium Comp Leasing Volumes:")
premium_properties = ['NOVEL Daybreak by Crescent Communities', 'Hamilton Crossing', 
                     'Parc Ridge', 'Solameer', 'Upper West', 'Soleil Lofts']

for prop in premium_properties:
    prop_leases = len(leased_df[leased_df['property_name'] == prop])
    if prop_leases > 0:
        multiple = prop_leases / district_total_leases if district_total_leases > 0 else 0
        print(f"  {prop}: {prop_leases} leases ({multiple:.1f}x District)")

print(f"\nTotal Premium Comp Leases: {sum([len(leased_df[leased_df['property_name'] == prop]) for prop in premium_properties])}")
total_premium_leases = sum([len(leased_df[leased_df['property_name'] == prop]) for prop in premium_properties])
district_market_share = district_total_leases / (district_total_leases + total_premium_leases) * 100
print(f"District's Share of Premium + District Market: {district_market_share:.1f}%")
