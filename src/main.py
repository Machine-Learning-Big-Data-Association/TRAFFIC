import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from mpl_toolkits.basemap import Basemap

from data import get_csv
from data import do_traffic_data
from data import do_driver_data

df = get_csv("sobhanmoosavi/us-accidents", "/US_Accidents_March23.csv", 1000000)
df = do_traffic_data(df)

drivers_df = pd.read_csv("data/Licensed_Drivers_by_State__Sex__and_Age_Group__1994_-_2023__DL-22_.csv")
drivers_df = do_driver_data(drivers_df)

# combines state and county
group_cols = ["State", "County"]
# defining how to aggregrate each data
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
# make a smaller data frame to hold data above
cols_to_keep = list(set(group_cols + list(agg_dict.keys())))
df_small = df[cols_to_keep].copy()
# groups smaller data set into counties and states
county_df = df_small.groupby(group_cols).agg(agg_dict).reset_index()
# aggregates the data according to previous aggregation rules
county_df = county_df.rename(columns={"ID": "Total_Accidents"})
county_df = county_df.dropna(subset=["Total_Accidents"])
# creates a collumn features 
feature_cols = [c for c in county_df.columns if c not in ["State", "County", "Total_Accidents"]]
# removes anything not containing a feature
county_df = county_df.dropna(subset=feature_cols)
print("County data ready:", county_df.shape)

# ============================================================
# Load Licensed Driver Data and Merge
# ============================================================

# drivers_df = load_and_prepare_driver_data("data/Licensed_Drivers_by_State__Sex__and_Age_Group__1994_-_2023__DL-22_.csv")
# Merge on state
county_df = county_df.merge(drivers_df, on="State", how="left")
# Optional: drop states with missing driver info
county_df = county_df.dropna(subset=["Total_Drivers"])
print("County data with driver counts:", county_df.shape)

# ============================================================
# 4. Train-test split
# ============================================================

X = county_df[feature_cols].values
y = county_df["Total_Accidents"].values
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, county_df.index, test_size=0.2, random_state=42
)

# ============================================================
# 5. Train Random Forest
# ============================================================

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_test = rf.predict(X_test)
print("R² on test:", r2_score(y_test, y_pred_test))
print("MAE on test:", mean_absolute_error(y_test, y_pred_test))

# ============================================================
# 6. Predict risk
# ============================================================

all_preds = rf.predict(X)
min_pred = all_preds.min()
max_pred = all_preds.max()
county_df["risk_score"] = 100 * (all_preds - min_pred) / (max_pred - min_pred + 1e-9)

# ============================================================
# 7. Top counties
# ============================================================

top_n = 20
top_risk = county_df.sort_values("risk_score", ascending=False).head(top_n)
print("\nTop 20 high-risk counties:")
print(top_risk[["State", "County", "Total_Accidents", "risk_score"]])

# ============================================================
# 8. Plot US bubble map with gradient circles
# ============================================================

# Compute mean lat/lon per county
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
     llcrnrlon=-125,  # moved a bit west
    llcrnrlat=20,    # moved a bit south to include Florida Keys
    urcrnrlon=-65,   # moved a bit east
    urcrnrlat=52,    # moved a bit north
    projection='lcc',
    lat_1=33, lat_2=45,
    lon_0=-95
)
m.drawcoastlines()
m.drawcountries()
m.drawstates()
x, y = m(plot_df["Start_Lng"].values, plot_df["Start_Lat"].values)
base_size = 150

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
plt.title("US Accident Risk Bubble Map with Gradient Circles")
plt.show()
