import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import map
import train
from dataCleaning import get_csv
from dataCleaning import do_traffic_data
from dataCleaning import do_driver_data

def main():
    # Load and clean accident data
    df = get_csv("sobhanmoosavi/us-accidents", "/US_Accidents_March23.csv")
    df = do_traffic_data(df)
       
    # Filter for year 2020 ONLY
    df = df[df["Start_Time"].dt.year == 2020]
    print(f"Using 2020 data only: {len(df)} accidents")

    # Skip the header rows and use proper column names
    drivers_df = pd.read_excel("data\counties-agegroup-2020.xlsx", skiprows=5)

    drivers_df = do_driver_data(drivers_df)
    
    # Define grouping and aggregation
    group_cols = ["State", "County"]
    agg_dict = {
        "ID": "count",
        "Severity": "mean",
        "Distance(mi)": "mean",
        "Temperature(F)": "mean",
        "Visibility(mi)": "mean",
        "Precipitation(in)": "mean",
        "Is_Night": "mean",
        "Is_Weekend": "mean"
    }

    county_df, feature_cols = train.clean(df, group_cols, agg_dict)
    
    # ============================================================
    # Merge with driver data (people 16+)
    # ============================================================

    # Standardize county names for better matching
    county_df['County_Clean'] = county_df['County'].str.strip().str.replace(' County', '', regex=False)
    drivers_df['County_Clean'] = drivers_df['County'].str.strip().str.replace(' County', '', regex=False)
    
    county_df['State_Clean'] = county_df['State'].str.strip()
    drivers_df['State_Clean'] = drivers_df['State'].str.strip()
    
    county_df = county_df.merge(
        drivers_df[['State_Clean', 'County_Clean', 'Total_People_16_plus', 'Total_Decade_Population_Change']], 
        left_on=['State_Clean', 'County_Clean'],
        right_on=['State_Clean', 'County_Clean'],
        how='left'
    )

    county_df = county_df.dropna(subset=["Total_People_16_plus", "Total_Decade_Population_Change"])
    
    # ============================================================
    # Calculate accidents per capita (per 1000 people 16+)
    # ============================================================
    
    if len(county_df) > 0:
        county_df["Accidents_Per_1000"] = (county_df["Total_Accidents"] / county_df["Total_People_16_plus"]) * 1000
    
    # Add population as a feature
    feature_cols_extended = feature_cols + ["Total_People_16_plus"]
    
    # ============================================================
    # Train-test split - predict accidents per capita
    # ============================================================

    X = county_df[feature_cols_extended].values
    y = county_df["Accidents_Per_1000"].values
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, county_df.index, test_size=0.2, random_state=42
    )
    
    # ============================================================
    # Train Random Forest
    # ============================================================
    print("\n=== Training Per-Capita Accident Model (2020 Data) ===")
    rf = train.train(200, 42, -1, X_train, y_train, X_test, y_test)
    
    # ============================================================
    # Predict risk (based on accidents per capita)
    # ============================================================
    print("\n=== Predicting Risk Scores ===")
    train.predictRisk(rf, county_df, X)
    
    # ============================================================
    # Feature Importance
    # ============================================================

    feature_importance = pd.DataFrame({
        'feature': feature_cols_extended,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Feature Importance (Per-Capita Model) ===")
    print(feature_importance)
    
    # ============================================================
    # Top counties by accidents per capita
    # ============================================================
    print("\n=== Top 20 High-Risk Counties (by accidents per 1000 people 16+) ===")
    top_n = 20
    top_risk = county_df.sort_values("Accidents_Per_1000", ascending=False).head(top_n)
    print(top_risk[["State", "County", "Total_Accidents", "Total_People_16_plus", "Accidents_Per_1000", "risk_score"]])
    
    # ============================================================
    # Bottom counties (safest)
    # ============================================================
    print("\n=== Bottom 20 Safest Counties (by accidents per 1000 people 16+) ===")
    bottom_risk = county_df.sort_values("Accidents_Per_1000", ascending=True).head(top_n)
    print(bottom_risk[["State", "County", "Total_Accidents", "Total_People_16_plus", "Accidents_Per_1000", "risk_score"]])
    
    # ============================================================
    # Predict 2030 using Linear Regression
    # ============================================================
    county_df = train.LinearRegression2030(county_df, feature_cols)
    
    # Add population as a feature
    feature_cols_extended = feature_cols + ["Population_Growth_2020_to_2030"]
    
    # ============================================================
    # Train-test split - predict accidents per capita
    # ============================================================

    X = county_df[feature_cols_extended].values
    y = county_df["Predicted_2030_Accidents_Total"].values
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, county_df.index, test_size=0.2, random_state=42
    )
    
    # ============================================================
    # Train Random Forest
    # ============================================================
    print("\n=== Training Per-Capita Accident Model (2020 Data) ===")
    rf = train.train(200, 42, -1, X_train, y_train, X_test, y_test)
    
    # ============================================================
    # Predict risk (based on accidents per capita)
    # ============================================================
    print("\n=== Predicting Risk Scores ===")
    train.predictRisk2030(rf, county_df, X)
    
    # ============================================================
    # Plot US bubble map
    # ============================================================
    print("\n=== Generating Map ===")
    map.makeYearlyAccidentsPer1000Line(df, county_df, title_suffix="")
    map.makeYearlyPredictionLinePopulation(df, county_df, title_suffix="")
    map.makeYearlyPredictionLine(df, county_df, title_suffix="")
    map.makeBarMap2030Accidents(df, county_df, title_suffix="")
    map.makeMap2030PopulationGrowthPercentage(df, county_df, title_suffix="")
    map.makeMap2030AccidentsGrowthPercentage(df, county_df, title_suffix="")
    map.makeStateMapAccidentGrowth(county_df, title_suffix="")
    map.makeMap2030RiskScore(df, county_df, title_suffix="")
    map.makeMap(df, county_df, title_suffix="")
    map.makeMapTotal(df, county_df, title_suffix="")
    map.makeMap2030Total(df, county_df, title_suffix="")
    map.makeMap2030AccidentsGrowthPercentage(df, county_df, title_suffix="")
    map.makeMap2030PopulationGrowth(df, county_df, title_suffix="")
    map.makeStateMapTotal(county_df, title_suffix="")
    map.makeStateMap2030Total(county_df, title_suffix="")
    map.makeStateMapPopulationGrowth(county_df, title_suffix="")
if __name__ == "__main__":
    main()
    