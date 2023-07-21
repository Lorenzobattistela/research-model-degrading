import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
)


def load_data(filename):
    df = pd.read_csv("stocks/" + filename + ".csv")
    return df


# Load the dataset
data = load_data("AAPL")

# Assuming the 'Date' column is in the correct datetime format, if not, convert it using:
data["Date"] = pd.to_datetime(data["Date"])

# Define the features and target variable
features = ["Close", "Volume", "Open", "High", "Low"]
target = "Close"


# Create a function to train the model, predict, and calculate metrics
def train_predict_metrics(train_start_year, train_end_year, test_year):
    # Filter data for the given training and testing periods
    train_data = data[
        (data["Date"] >= f"{train_start_year}-01-01")
        & (data["Date"] <= f"{train_end_year}-12-31")
    ]
    test_data = data[data["Date"].dt.year == test_year]

    # Extract the features and target variable for training and testing
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # Perform polynomial regression with degree=2
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    explained_var = explained_variance_score(y_test, y_pred)

    return {
        "year-trained": train_year,
        "year-predicted": predict_year,
        "mse": mse,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "explained_var": explained_var,
    }


# Create an empty list to store the results
results = []

# Loop through the years to train, predict, and store the results
train_years = np.arange(1999, 2019)  # Train from 1999 to 2018
predict_years = np.arange(2013, 2021)  # Predict from 2013 to 2020

for train_year in train_years:
    for predict_year in predict_years:
        # Skip if the training year is greater than or equal to the testing year
        if train_year >= predict_year:
            continue

        # Train, predict, and calculate metrics
        metrics = train_predict_metrics(train_year, train_year + 13, predict_year)

        # Store the metrics in the list
        results.append(metrics)

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results)

# Save the results to a CSV file
results_df.to_csv("polynomial_regression_results.csv", index=False)

print("Analysis completed. Results saved to 'polynomial_regression_results.csv'.")
