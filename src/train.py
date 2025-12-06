import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def train(nEsimator, randomState, nJobs, xTrain, yTrain, xTest, yTest):
    attempt = 1
    max_attempts = 5

    params = {
        "n_estimators": nEsimator,
        "random_state": randomState,
        "n_jobs": nJobs,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }

    best_model = None

    while attempt <= max_attempts:
        print(f"\n===== Training Attempt {attempt}/{max_attempts} =====")
        print("Using parameters:", params)

        rf = RandomForestRegressor(**params)
        rf.fit(xTrain, yTrain)

        y_pred_test = rf.predict(xTest)
        y_pred_train = rf.predict(xTrain)

        r2_train = r2_score(yTrain, y_pred_train)
        r2_test = r2_score(yTest, y_pred_test)

        print(f"Train R²: {r2_train:.4f}")
        print(f"Test  R²: {r2_test:.4f}")
        print(f"Train MAE: {mean_absolute_error(yTrain, y_pred_train):.4f}")
        print(f"Test  MAE: {mean_absolute_error(yTest, y_pred_test):.4f}")

        overfit = (r2_train >= 0.90) and ((r2_train - r2_test) >= 0.25)

        if not overfit:
            print("Model passed overfitting check.")
            best_model = rf
            break

        print("!!! OVERFITTING DETECTED — lowering model complexity...")

        # AUTO-LOWER MODEL COMPLEXITY
        params["n_estimators"] = max(20, params["n_estimators"] // 2)
        params["max_depth"] = 15 if params["max_depth"] is None else max(5, params["max_depth"] - 5)
        params["min_samples_split"] = min(params["min_samples_split"] + 1, 10)
        params["min_samples_leaf"] = min(params["min_samples_leaf"] + 1, 10)
        params["max_features"] = "sqrt"

        attempt += 1

    if best_model is None:
        print("!!! All attempts overfit. Returning simplest model.")
        best_model = rf

    return best_model


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
def predictRisk(rf, county_df, X):
    all_preds = rf.predict(X)
    min_pred = all_preds.min()
    max_pred = all_preds.max()
    norm_preds = (all_preds - min_pred) / (max_pred - min_pred + 1e-9)
    norm_preds = np.sqrt(norm_preds)
    county_df["risk_score"] = norm_preds * 10
    return None

def predictRisk2030(rf, county_df, X):
    all_preds = rf.predict(X)
    min_pred = all_preds.min()
    max_pred = all_preds.max()
    norm_preds = (all_preds - min_pred) / (max_pred - min_pred + 1e-9)
    norm_preds = np.sqrt(norm_preds)
    county_df["risk_score_2030"] = norm_preds * 10
    return None

def LinearRegression2030(county_df, feature_cols):
    county_df["Projected_2030_Population_16_plus"] = (
        county_df["Total_People_16_plus"] + 
        county_df["Total_Decade_Population_Change"]
    )

    # Make sure projected population is not negative
    county_df["Projected_2030_Population_16_plus"] = county_df["Projected_2030_Population_16_plus"].clip(lower=0)

    print("Training Linear Regression model for 2030 predictions...")

    # Create features for training (using 2020 data)
    X_train_lr = np.column_stack([
        county_df[feature_cols].values,
        county_df["Total_People_16_plus"].values
    ])

    # Target: accidents per capita in 2020
    y_train_lr = county_df["Accidents_Per_1000"].values

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)

    # Check model performance on training data
    y_pred_train_lr = lr_model.predict(X_train_lr)
    print(f"Linear Regression R² on training data: {r2_score(y_train_lr, y_pred_train_lr):.4f}")
    print(f"Linear Regression MAE on training data: {mean_absolute_error(y_train_lr, y_pred_train_lr):.4f}")

    # 3. Create features for 2030 prediction
    # Use same environmental features but with projected population
    X_2030 = np.column_stack([
        county_df[feature_cols].values,
        county_df["Projected_2030_Population_16_plus"].values
    ])

    # Predict per-capita accidents for 2030 using Linear Regression
    # This gives us accidents per 1000 people
    predicted_per_capita_2030 = lr_model.predict(X_2030)

    # Ensure predictions are non-negative
    predicted_per_capita_2030 = np.maximum(predicted_per_capita_2030, 0)

    # Store the per-capita prediction
    county_df["Predicted_2030_Accidents_PerCapita"] = predicted_per_capita_2030

    # Convert to total accidents
    # Formula: (accidents per 1000) * (population / 1000)
    county_df["Predicted_2030_Accidents_Total"] = (
        predicted_per_capita_2030 * 
        (county_df["Projected_2030_Population_16_plus"] / 1000)
    )

    # 4. Calculate growth metrics
    county_df["Accident_Growth_2020_to_2030"] = (
        county_df["Predicted_2030_Accidents_Total"] - county_df["Total_Accidents"]
    )

    county_df["Accident_Growth_Percentage"] = (
    county_df["Accident_Growth_2020_to_2030"] /
    ((county_df["Predicted_2030_Accidents_Total"] + county_df["Total_Accidents"]) / 2)
    ) * 100


    county_df["Population_Growth_2020_to_2030"] = (
        county_df["Projected_2030_Population_16_plus"] - county_df["Total_People_16_plus"]
    )

    county_df["Population_Growth_Percentage"] = (
        (county_df["Population_Growth_2020_to_2030"] / county_df["Total_People_16_plus"]) * 100
    )

    # 5. Print Linear Regression coefficients for interpretation
    print("\n=== Linear Regression Coefficients ===")
    feature_names_extended = feature_cols + ["Total_People_16_plus"]
    coef_df = pd.DataFrame({
        'Feature': feature_names_extended,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df)
    print(f"\nIntercept: {lr_model.intercept_:.4f}")
    
    return county_df