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
    """Clean and standardize state and county names"""
    # Strip whitespace from State and County columns
    if "State" in df.columns:
        df["State"] = df["State"].str.strip()
    if "County" in df.columns:
        df["County"] = df["County"].str.strip()
        # Remove " County" suffix if present (e.g., "Los Angeles County" -> "Los Angeles")
        df["County"] = df["County"].str.replace(" County", "", regex=False)
    
    return df

def normalize_Abbreviations(df):
    """Convert state abbreviations to full names if needed"""
    state_abbrev_to_full = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
    }
    
    # Also create reverse mapping (full name to abbreviation)
    state_full_to_abbrev = {v: k for k, v in state_abbrev_to_full.items()}
    
    # Check if State column contains abbreviations (2 characters) or full names
    if "State" in df.columns:
        sample_state = str(df["State"].iloc[0]) if len(df) > 0 else ""
        if len(sample_state) == 2:
            # Convert abbreviations to full names
            df["State"] = df["State"].map(state_abbrev_to_full).fillna(df["State"])
            print(f"Converted state abbreviations to full names")
        else:
            # Already full names, no conversion needed
            print(f"State names are already in full form")
    
    return df

# returns a df of "State", "County", "Total_People_16_plus"
def combine_drivers_data(df):
    """
    Process driver data to calculate people 16+ by county
    Handles the specific Excel format with age group columns
    """
    print(f"Processing driver data with shape: {df.shape}")
    
    # The Excel has multi-level headers. The age groups are in the column names already
    # We need to get State and County from the first row of data
    
    # First, identify which columns have the age group names we want
    age_group_columns = {
        '18 to 24 years': None,
        '25 to 34 years': None,
        '35 to 44 years': None,
        '45 to 64 years': None,
        '65 to 84 years': None,
        '85 to 99 years': None,
        '100 years and over': None
    }
    
    # Find the column index for each age group
    for i, col in enumerate(df.columns):
        if col in age_group_columns:
            age_group_columns[col] = i
            print(f"Found '{col}' at column index {i}")
    
    # The first row contains the actual column headers (State, County, etc.)
    header_row = df.iloc[0].tolist()
    print(f"\nHeader row: {header_row[:10]}...")  # Show first 10
    
    # Find State and County column indices
    state_idx = header_row.index('State') if 'State' in header_row else None
    county_idx = header_row.index('County') if 'County' in header_row else None
    
    print(f"State column index: {state_idx}")
    print(f"County column index: {county_idx}")
    
    if state_idx is None or county_idx is None:
        print("ERROR: Could not find State or County in header row")
        return pd.DataFrame()
    
    # Skip first 2 rows (header info) and start with actual data
    df_data = df.iloc[2:].copy()
    df_data = df_data.reset_index(drop=True)
    
    # Get the original column names (which have the age groups)
    orig_cols = df.columns.tolist()
    
    # Extract State and County using their indices
    state_col = orig_cols[state_idx]
    county_col = orig_cols[county_idx]
    
    print(f"\nUsing column '{state_col}' for State")
    print(f"Using column '{county_col}' for County")
    
    # Rename for clarity
    df_data = df_data.rename(columns={
        state_col: 'State',
        county_col: 'County'
    })
    
    # Now extract the age columns (the ones with population counts)
    # For each age group, we want the "Population" column under it
    age_cols_to_use = []
    for age_group, idx in age_group_columns.items():
        if idx is not None:
            # The population count is in the same column as the age group name
            col_name = orig_cols[idx]
            age_cols_to_use.append(col_name)
            # Rename it for clarity
            df_data = df_data.rename(columns={col_name: f'{age_group}_pop'})
    
    print(f"\nAge columns to sum: {[f'{ag}_pop' for ag in age_group_columns.keys() if age_group_columns[ag] is not None]}")
    
    # Convert to numeric
    renamed_age_cols = [f'{ag}_pop' for ag in age_group_columns.keys() if age_group_columns[ag] is not None]
    for col in renamed_age_cols:
        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
    
    # Sum to get 16+
    df_data["People_16_plus"] = df_data[renamed_age_cols].sum(axis=1)
    
    print(f"\nData sample after processing:")
    print(df_data[['State', 'County', 'People_16_plus']].head(10))
    
    # Group by State and County
    drivers_county = (
        df_data.groupby(['State', 'County'])["People_16_plus"]
        .sum()
        .reset_index()
        .rename(columns={"People_16_plus": "Total_People_16_plus"})
    )
    
    # Remove any rows with zero or NaN population
    drivers_county = drivers_county[drivers_county["Total_People_16_plus"] > 0]
    drivers_county = drivers_county.dropna(subset=["State", "County"])
    
    # Clean up state and county names
    drivers_county["State"] = drivers_county["State"].astype(str).str.strip()
    drivers_county["County"] = drivers_county["County"].astype(str).str.strip()
    
    # Remove any 'nan' or empty strings
    drivers_county = drivers_county[
        (drivers_county["State"] != 'nan') & 
        (drivers_county["State"] != '') &
        (drivers_county["County"] != 'nan') & 
        (drivers_county["County"] != '')
    ]
    
    print(f"\nDriver data prepared: {drivers_county.shape}")
    print(f"Sample of processed data:")
    print(drivers_county.head(10))
    print(f"\nPopulation 16+ statistics:")
    print(drivers_county["Total_People_16_plus"].describe())
    
    return drivers_county

def clean_cars_data(df):
    df = df[df["Count"] > 0]
    df = df.dropna(subset=["Transaction County","Residential County", "Count"])
    return df

def do_driver_data(df):
    df = clean_drivers_data(df)
    df = normalize_Abbreviations(df)
    df = combine_drivers_data(df)
    return df

def do_traffic_data(df):
    df = clean_data(df)
    df = add_data(df)
    df = normalize_Abbreviations(df)  # Add this to normalize state names
    return df

def do_cars_data(df):
    df = clean_cars_data(df)
    return df