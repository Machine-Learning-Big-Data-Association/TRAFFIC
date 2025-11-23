import pandas as pd
import kagglehub
from datetime import datetime
# ALL DATA CLEANING AND ADDING IN HERE

# gets the data from kaggle using the path and csv
def get_data(path, csv, rows):    
    # gets the data set from kagglehub
    path = kagglehub.dataset_download(path)
    # stores it into dataframe called df
    df = pd.read_csv(path + csv, nrows= rows)
    # df = pd.read_csv(path + csv)
    return df

# cleans "Start_Time" and "End_Time"
def time_data(df):
    # attempts to convert data into date time format deletes if unable to
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.loc[df["Start_Time"].notna()]

    # Probably not needed ?? VVVVV
    # b/c we alr convert start time and just use that?? either way cleans out any unreadable end times

    # attempts to convert data into date time format deletes if unable to
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    df = df.loc[df["End_Time"].notna()]
    return df

# cleans Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", and "Precipitation(in)"
def clean_num_data(df):
    # cleans out data by removing any non numerical data in these collumns
    num_cols = ["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"]
    for c in num_cols:
        if c in df.columns:
            df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

    # the actual delete            
    df = df.dropna(subset=["Severity", "Distance(mi)", "Temperature(F)", "Visibility(mi)", "Precipitation(in)"])
    return df

# does all cleans above and cleans "County", "State", "Start_Lat", and "Start_Lng"
def clean_data(df):
    # remove rows if data dont exist
    df = df.dropna(subset=["County", "State", "Start_Lat", "Start_Lng"])
    df = time_data(df)
    df = clean_num_data(df)
    return df
    
# Adds "Hour", "Is_Night", "Day_of_Week", and "Is_Weekend"
def add_data(df):
    # gets hours and makes a new column from start time
    df.loc[:, "Hour"] = df["Start_Time"].dt.hour
    df.loc[:, "Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    # is true (1) when hours are before 6am (exclusive) or after 8pm (inclusive)
    # important when driving because less visability
    df.loc[:, "Is_Night"] = ((df["Hour"] < 6) | (df["Hour"] >= 20)).astype(int)
    # gets which day of the week stored as 0-6
    df.loc[:, "Day_of_Week"] = df["Start_Time"].dt.dayofweek
    # checks if day of week is 5(saturday) or 6(sunday)
    # more recreational travel and less morning/evening rush hour
    df.loc[:, "Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    return df

def load_and_prepare_driver_data(csv_path):
    """
    Takes the full drivers dataset (with Year, Sex, Cohort, etc.)
    and returns a dataframe with:
    
    State | Licensed_Drivers_Total
    """
    df = pd.read_csv(csv_path)

    # Clean state column
    df["State"] = df["State"].str.strip()

    # Group by state and sum drivers across all ages/sexes
    drivers_state = (
        df.groupby("State")["Drivers"]
        .sum()
        .reset_index()
        .rename(columns={"Drivers": "Licensed_Drivers"})
    )

    return drivers_state

# does everything
def do_data(df):
    df = clean_data(df)
    df = add_data(df)
    return df

