import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from mpl_toolkits.basemap import Basemap
from dataCleaning import get_csv
from dataCleaning import do_traffic_data
from dataCleaning import do_driver_data
from dataCleaning import do_cars_data

def train(nEsimator, randomState, nJobs, xTrain, yTrain, xTest, yTest):
    rf = RandomForestRegressor(n_estimators=nEsimator, random_state=randomState, n_jobs=nJobs)
    rf.fit(xTrain, yTrain)
    y_pred_test = rf.predict(xTest)
    print("R² on test:", r2_score(yTest, y_pred_test))
    print("MAE on test:", mean_absolute_error(yTest, y_pred_test))
    return rf

def makeMap(df, county_df, title_suffix=""):
    county_coords = df.groupby(["State", "County"]).agg({
        "Start_Lat": "mean",
        "Start_Lng": "mean"
    }).reset_index()
    
    plot_df = county_df.merge(county_coords, on=["State", "County"])
    
    # Handle extreme outliers
    vmax = np.percentile(plot_df["risk_score"], 99)
    colors = np.clip(plot_df["risk_score"], None, vmax)
    
    plt.figure(figsize=(14,8))
    m = Basemap(
        llcrnrlon=-125,
        llcrnrlat=20,
        urcrnrlon=-65,
        urcrnrlat=52,
        projection='lcc',
        lat_1=33,
        lat_2=45,
        lon_0=-95
    )
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawcounties()
    m.drawstates()
    
    x, y = m(plot_df["Start_Lng"].values, plot_df["Start_Lat"].values)
    base_size = 50
    
    # Normalize risk for colormap
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.Reds
    colors_mapped = cmap(norm(colors))
    
    ax = plt.gca()
    
    # Plot gradient circles
    for i in range(len(x)):
        for alpha, scale in zip([0.4, 0.2, 0.1], [1.0, 1.5, 2.0]):
            m.scatter(
                x[i], y[i],
                s=base_size * scale,
                color=colors_mapped[i],
                alpha=alpha,
                edgecolors='none'
            )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label="Risk Score (0-100, clipped at 99th percentile)")
    
    plt.title(f"US Accident Risk Per Capita {title_suffix}")
    plt.show()
    return None

def clean(df, group_cols, agg_dict):
    # make a smaller data frame to hold data above cols
    cols_to_keep = list(set(group_cols + list(agg_dict.keys())))
    df_small = df[cols_to_keep].copy()
    
    # groups smaller data set into counties and states
    county_df = df_small.groupby(group_cols).agg(agg_dict).reset_index()
    
    # aggregates the data according to previous aggregation rules
    county_df = county_df.rename(columns={"ID": "Total_Accidents"})
    county_df = county_df.dropna(subset=["Total_Accidents"])
    
    # creates a column features
    feature_cols = [c for c in county_df.columns if c not in ["State", "County", "Total_Accidents"]]
    
    # removes anything not containing a feature
    county_df = county_df.dropna(subset=feature_cols)
    print("County data ready:", county_df.shape)
    
    return county_df, feature_cols

def main():
    # Load and clean accident data
    df = get_csv("sobhanmoosavi/us-accidents", "/US_Accidents_March23.csv", 10000000)  # Increase to 100k for better coverage
    df = do_traffic_data(df)
    
    print(f"\nTotal accidents loaded: {len(df)}")
    print(f"Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
    print(f"Years present: {sorted(df['Start_Time'].dt.year.unique())}")
    
    # Filter for year 2020
    df2020 = df[df["Start_Time"].dt.year == 2020]
    print(f"2020 accidents: {len(df2020)}")
    
    # Load driver data (people 16+)
    print("\n=== Loading Driver Data ===")
    # Skip the header rows and use proper column names
    drivers_df = pd.read_excel("TRAFFIC/data/counties-agegroup-2020.xlsx", skiprows=5)
    print(f"Raw driver data shape: {drivers_df.shape}")
    print(f"Columns: {drivers_df.columns.tolist()}")
    print(f"First few rows:\n{drivers_df.head()}")
    drivers_df = do_driver_data(drivers_df)
    
    # Load car registration data (optional - not used yet)
    # cars_df = pd.read_csv("TRAFFIC/data/Vehicle_Registrations_by_Class_and_County.csv")
    # cars_df = do_cars_data(cars_df)
    
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
    
    print("\n=== Processing All Years Data ===")
    county_df, feature_cols = clean(df, group_cols, agg_dict)
    
    print("\n=== Processing 2020 Data ===")
    county_df2020, feature2020_cols = clean(df2020, group_cols, agg_dict)
    
    # ============================================================
    # Merge with driver data (people 16+)
    # ============================================================
    print("\n=== Merging with Population Data ===")
    
    # Debug: Check sample county names from both datasets
    print("\nSample accident data counties:")
    print(county_df[['State', 'County']].head(10))
    
    print("\nSample driver data counties:")
    print(drivers_df[['State', 'County']].head(10))
    
    # Standardize county names for better matching
    county_df['County_Clean'] = county_df['County'].str.strip().str.replace(' County', '', regex=False)
    drivers_df['County_Clean'] = drivers_df['County'].str.strip().str.replace(' County', '', regex=False)
    
    county_df['State_Clean'] = county_df['State'].str.strip()
    drivers_df['State_Clean'] = drivers_df['State'].str.strip()
    
    county_df = county_df.merge(
        drivers_df[['State_Clean', 'County_Clean', 'Total_People_16_plus']], 
        left_on=['State_Clean', 'County_Clean'],
        right_on=['State_Clean', 'County_Clean'],
        how='left'
    )
    
    print(f"After merge: {county_df.shape}")
    print(f"Missing population data: {county_df['Total_People_16_plus'].isna().sum()} counties")
    
    # Show which counties didn't match
    unmatched = county_df[county_df['Total_People_16_plus'].isna()][['State', 'County']]
    if len(unmatched) > 0:
        print(f"\nUnmatched counties (first 10):")
        print(unmatched.head(10))
    
    county_df = county_df.dropna(subset=["Total_People_16_plus"])
    print(f"After dropping NAs: {county_df.shape}")
    
    # Same for 2020
    county_df2020['County_Clean'] = county_df2020['County'].str.strip().str.replace(' County', '', regex=False)
    county_df2020['State_Clean'] = county_df2020['State'].str.strip()
    
    county_df2020 = county_df2020.merge(
        drivers_df[['State_Clean', 'County_Clean', 'Total_People_16_plus']], 
        left_on=['State_Clean', 'County_Clean'],
        right_on=['State_Clean', 'County_Clean'],
        how='left'
    )
    county_df2020 = county_df2020.dropna(subset=["Total_People_16_plus"])
    
    # ============================================================
    # Calculate accidents per capita (per 1000 people 16+)
    # ============================================================
    if len(county_df) > 0:
        print("\n=== Calculating Per Capita Accident Rates ===")
        county_df["Accidents_Per_1000"] = (county_df["Total_Accidents"] / county_df["Total_People_16_plus"]) * 1000
        
        print(f"Accidents per 1000 range: {county_df['Accidents_Per_1000'].min():.2f} to {county_df['Accidents_Per_1000'].max():.2f}")
        print(f"Mean accidents per 1000: {county_df['Accidents_Per_1000'].mean():.2f}")
    else:
        print("\n=== ERROR: No counties with population data ===")
        print("Cannot continue with analysis. Please check county name matching.")
        return
    
    if len(county_df2020) > 0:
        county_df2020["Accidents_Per_1000"] = (county_df2020["Total_Accidents"] / county_df2020["Total_People_16_plus"]) * 1000
    
    # Add population as a feature
    feature_cols_extended = feature_cols + ["Total_People_16_plus"]
    
    # ============================================================
    # Train-test split - predict accidents per capita
    # ============================================================
    print("\n=== Training Model ===")
    X = county_df[feature_cols_extended].values
    y = county_df["Accidents_Per_1000"].values
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, county_df.index, test_size=0.2, random_state=42
    )
    
    # ============================================================
    # Train Random Forest
    # ============================================================
    rf = train(200, 42, -1, X_train, y_train, X_test, y_test)
    
    # ============================================================
    # Predict risk (based on accidents per capita)
    # ============================================================
    print("\n=== Generating Risk Scores ===")
    all_preds = rf.predict(X)
    min_pred = all_preds.min()
    max_pred = all_preds.max()
    county_df["risk_score"] = 100 * (all_preds - min_pred) / (max_pred - min_pred + 1e-9)
    
    # Same for 2020 data (only if we have data)
    if len(county_df2020) > 0:
        X_2020 = county_df2020[feature_cols_extended].values
        all_preds_2020 = rf.predict(X_2020)
        min_pred_2020 = all_preds_2020.min()
        max_pred_2020 = all_preds_2020.max()
        county_df2020["risk_score"] = 100 * (all_preds_2020 - min_pred_2020) / (max_pred_2020 - min_pred_2020 + 1e-9)
    else:
        print("Skipping 2020 risk scores - no 2020 data available")
    
    # ============================================================
    # Feature Importance
    # ============================================================
    print("\n=== Feature Importance ===")
    feature_importance = pd.DataFrame({
        'feature': feature_cols_extended,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
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
    # Plot US bubble maps
    # ============================================================
    print("\n=== Generating Maps ===")
    makeMap(df, county_df, title_suffix="(All Years)")
    
    if len(county_df2020) > 0 and len(df2020) > 0:
        makeMap(df2020, county_df2020, title_suffix="(2020 Only)")
    else:
        print("Skipping 2020 map - insufficient 2020 data")

if __name__ == "__main__":
    main()
    