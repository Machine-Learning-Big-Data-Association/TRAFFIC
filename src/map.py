import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.basemap import Basemap
import geopandas as gpd


# Helper: apply outlier clipping using percentile
def percentile_norm(series, p):
    clean = series.dropna()
    if clean.empty:
        return colors.Normalize(vmin=0, vmax=1)
    vmax = np.percentile(clean, p)
    return colors.Normalize(vmin=clean.min(), vmax=vmax)


def load_counties():
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_5m.zip"
    counties = gpd.read_file(url)

    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }

    counties["County_Match"] = counties["NAME"].str.upper().str.strip()
    counties["State_Match"] = counties["STATEFP"].map(state_fips)

    # Drop AK, HI, PR
    return counties[~counties["STATEFP"].isin(["02", "15", "72"])]


cmap = colors.LinearSegmentedColormap.from_list(
    "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
)


def plot_map(counties, value_column, title_text, p):
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)

    # Base map
    counties.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.0)

    # Filter for mapped data
    filled = counties[counties[value_column].notna()]

    # Normalize using percentile clipping
    norm = percentile_norm(filled[value_column], p)

    filled.plot(
        ax=ax,
        column=value_column,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.4,
        legend=False
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(value_column.replace("_", " "), fontsize=16, fontweight="bold")

    # Set extent
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()

    plt.title(title_text, fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# -------------------------
# MAP FUNCTION REWRITES
# -------------------------

def makeMap(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "risk_score"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "risk_score",
             f"US Accident Risk by County {title_suffix}", 100)


def makeMapTotal(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper()
    plot_df["State_Match"] = plot_df["State"].str.upper()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "Total_Accidents"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "Total_Accidents",
             f"US Total Accidents (2020) by County {title_suffix}", 97)


def makeMap2030Total(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper()
    plot_df["State_Match"] = plot_df["State"].str.upper()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "Predicted_2030_Accidents_Total"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "Predicted_2030_Accidents_Total",
             f"Predicted Accident Totals (2030) {title_suffix}", 97)


def makeMap2030AccidentsGrowthPercentage(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper()
    plot_df["State_Match"] = plot_df["State"].str.upper()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "Accident_Growth_Percentage"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "Accident_Growth_Percentage",
             f"Predicted Accident Growth % (2030) {title_suffix}", 90)
    
def makeMap2030PopulationGrowth(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper()
    plot_df["State_Match"] = plot_df["State"].str.upper()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "Population_Growth_2020_to_2030"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "Population_Growth_2020_to_2030",
             f"Predicted Driver Growth (2030) {title_suffix}", 95)
    
def makeMap2030RiskScore(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper().str.strip()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "risk_score_2030"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "risk_score_2030",
             f"US Accident Risk by County (2030) {title_suffix}", 99)



def makeMap2030PopulationGrowthPercentage(df, county_df, title_suffix=""):
    counties = load_counties()

    plot_df = county_df.copy()
    plot_df["County_Match"] = plot_df["County"].str.upper()
    plot_df["State_Match"] = plot_df["State"].str.upper()

    merged = counties.merge(
        plot_df[["State_Match", "County_Match", "Population_Growth_Percentage"]],
        on=["State_Match", "County_Match"],
        how="left"
    )

    plot_map(merged, "Population_Growth_Percentage",
             f"Predicted Driver Growth % (2030) {title_suffix}", 100)

def load_states():
    """Load US state boundaries from Census Bureau"""
    url = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_5m.zip"
    states = gpd.read_file(url)
    
    state_fips = {
        '01':'ALABAMA','02':'ALASKA','04':'ARIZONA','05':'ARKANSAS','06':'CALIFORNIA',
        '08':'COLORADO','09':'CONNECTICUT','10':'DELAWARE','11':'DISTRICT OF COLUMBIA',
        '12':'FLORIDA','13':'GEORGIA','15':'HAWAII','16':'IDAHO','17':'ILLINOIS',
        '18':'INDIANA','19':'IOWA','20':'KANSAS','21':'KENTUCKY','22':'LOUISIANA',
        '23':'MAINE','24':'MARYLAND','25':'MASSACHUSETTS','26':'MICHIGAN','27':'MINNESOTA',
        '28':'MISSISSIPPI','29':'MISSOURI','30':'MONTANA','31':'NEBRASKA','32':'NEVADA',
        '33':'NEW HAMPSHIRE','34':'NEW JERSEY','35':'NEW MEXICO','36':'NEW YORK',
        '37':'NORTH CAROLINA','38':'NORTH DAKOTA','39':'OHIO','40':'OKLAHOMA','41':'OREGON',
        '42':'PENNSYLVANIA','44':'RHODE ISLAND','45':'SOUTH CAROLINA','46':'SOUTH DAKOTA',
        '47':'TENNESSEE','48':'TEXAS','49':'UTAH','50':'VERMONT','51':'VIRGINIA',
        '53':'WASHINGTON','54':'WEST VIRGINIA','55':'WISCONSIN','56':'WYOMING','72':'PUERTO RICO'
    }
    
    states["State_Match"] = states["STATEFP"].map(state_fips)
    
    # Drop AK, HI, PR for continental US
    return states[~states["STATEFP"].isin(["02", "15", "72"])]


cmap = colors.LinearSegmentedColormap.from_list(
    "risk_gradient", ["#a1d99b", "#fdae61", "#d73027"]
)


def plot_state_map(states, value_column, title_text, cbar_label, p):
    """Generic function to plot state-level choropleth maps"""
    fig, ax = plt.subplots(figsize=(18, 10), dpi=60)
    
    # Base map
    states.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=1.5)
    
    # Filter for states with data
    filled = states[states[value_column].notna()]
    
    # Normalize using percentile clipping
    norm = percentile_norm(filled[value_column], p)
    
    filled.plot(
        ax=ax,
        column=value_column,
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.8,
        legend=False
    )
    
    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(cbar_label, fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)
    
    # Set extent for continental US
    ax.set_xlim(-130, -65)
    ax.set_ylim(22, 52)
    ax.set_axis_off()
    
    plt.title(title_text, fontsize=20, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# -------------------------
# STATE-LEVEL MAP FUNCTIONS
# -------------------------

def makeStateMapTotal(county_df, title_suffix=""):
    """Map total accidents by state for 2020"""
    states = load_states()
    
    # Aggregate county data to state level
    plot_df = county_df.copy()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    
    state_df = plot_df.groupby("State_Match").agg({
        "Total_Accidents": "sum"
    }).reset_index()
    
    # Merge with state boundaries
    merged = states.merge(
        state_df,
        on="State_Match",
        how="left"
    )
    
    avg_accidents = state_df["Total_Accidents"].mean()
    
    plot_state_map(
        merged, 
        "Total_Accidents",
        f"US Total Accidents (2020) by State {title_suffix}\n"
        f"{len(state_df)} states with data | "
        f"Avg: {avg_accidents:,.0f} accidents per state",
        "Total Accidents",
        p=97
    )


def makeStateMap2030Total(county_df, title_suffix=""):
    states = load_states()
    
    # Aggregate county data to state level
    plot_df = county_df.copy()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    
    state_df = plot_df.groupby("State_Match").agg({
        "Predicted_2030_Accidents_Total": "sum"
    }).reset_index()
    
    # Merge with state boundaries
    merged = states.merge(
        state_df,
        on="State_Match",
        how="left"
    )
    
    avg_accidents = state_df["Predicted_2030_Accidents_Total"].mean()
    
    plot_state_map(
        merged,
        "Predicted_2030_Accidents_Total",
        f"Predicted US Accidents (2030) by State {title_suffix}\n"
        f"{len(state_df)} states with data | "
        f"Avg: {avg_accidents:,.0f} predicted accidents per state",
        "Predicted Total Accidents",
        p=97
    )


def makeStateMapAccidentGrowth(county_df, title_suffix=""):
    """Map accident growth percentage by state (2020 to 2030)"""
    states = load_states()
    
    # Aggregate county data to state level
    plot_df = county_df.copy()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    
    state_df = plot_df.groupby("State_Match").agg({
        "Total_Accidents": "sum",
        "Predicted_2030_Accidents_Total": "sum"
    }).reset_index()
    
    # Calculate growth percentage at state level
    state_df["Accident_Growth_Percentage"] = (
        (state_df["Predicted_2030_Accidents_Total"] - state_df["Total_Accidents"]) / 
        state_df["Total_Accidents"]
    ) * 100
    
    # Merge with state boundaries
    merged = states.merge(
        state_df[["State_Match", "Accident_Growth_Percentage"]],
        on="State_Match",
        how="left"
    )
    
    avg_growth = state_df["Accident_Growth_Percentage"].mean()
    
    plot_state_map(
        merged,
        "Accident_Growth_Percentage",
        f"Predicted Accident Growth % (2020-2030) by State {title_suffix}\n"
        f"{len(state_df)} states with data | "
        f"Avg: {avg_growth:.1f}% growth",
        "Accident Growth %",
        p=95
    )


def makeStateMapPopulationGrowth(county_df, title_suffix=""):
    states = load_states()
    
    # Aggregate county data to state level
    plot_df = county_df.copy()
    plot_df["State_Match"] = plot_df["State"].str.upper().str.strip()
    
    state_df = plot_df.groupby("State_Match").agg({
        "Total_People_16_plus": "sum",
        "Projected_2030_Population_16_plus": "sum"
    }).reset_index()
    
    # Calculate growth percentage at state level
    state_df["Population_Growth_Percentage"] = (
        (state_df["Projected_2030_Population_16_plus"] - state_df["Total_People_16_plus"]) / 
        state_df["Total_People_16_plus"]
    ) * 100
    
    # Merge with state boundaries
    merged = states.merge(
        state_df[["State_Match", "Population_Growth_Percentage"]],
        on="State_Match",
        how="left"
    )
    
    avg_growth = state_df["Population_Growth_Percentage"].mean()
    
    plot_state_map(
        merged,
        "Population_Growth_Percentage",
        f"Predicted Driver Growth % (2020-2030) by State {title_suffix}\n"
        f"{len(state_df)} states with data | "
        f"Avg: {avg_growth:.1f}% growth",
        "Driver Growth %",
        p=100
    )