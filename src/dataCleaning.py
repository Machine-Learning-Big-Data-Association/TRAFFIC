import pandas as pd
import kagglehub
from datetime import datetime
# ALL DATA CLEANING AND ADDING IN HERE

# gets the data from kaggle using the path and csv
def get_csv(path, csv, rows):    
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

# cleans "State" data
def clean_drivers_data(df):
    df["State"] = df["State"].str.strip()
    df = normalize_Abbreviations(df)
    return df

def normalize_Abbreviations(df):
    state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}
    # Normalize the State column
    df["State"] = df["State"].map(state_abbrev)
    return df
# returns a df of "State", "Total_Drivers"
def combine_drivers_data(df):
    drivers_state = (
        df.groupby("State")["Drivers"]
        .sum()
        .reset_index()
        .rename(columns={"Drivers": "Total_Drivers"})
    )
    return drivers_state

def do_driver_data(df):
    df = clean_drivers_data(df)
    df = combine_drivers_data(df)
    return df
def do_traffic_data(df):
    df = clean_data(df)
    df = add_data(df)
    return df