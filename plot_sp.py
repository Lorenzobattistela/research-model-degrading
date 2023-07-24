import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the CSV file
data = pd.read_csv("sp500_polynomial_regression_results.csv")

# Calculate the time difference between training and prediction years
data["time_difference"] = data["year-predicted"] - data["year-trained"]


# Function to plot performance difference vs. time difference and save the figure
def plot_and_save_performance_diff(data, metric, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x="time_difference", y=metric)
    plt.xlabel("Time Difference (Years)")
    plt.ylabel(f"{metric} Difference")
    plt.axhline(y=0, color="red", linestyle="--")  # Reference line at y=0 (no change)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"sp500_metrics_performance_{metric}_vs_year_difference.png")
    plt.close()


# Plot 1 - Mean Squared Error (MSE) Difference vs. Time Difference
plot_and_save_performance_diff(
    data, "mse", "Model Performance Difference - Mean Squared Error (MSE)"
)

# Plot 2 - R-squared (R2) Difference vs. Time Difference
plot_and_save_performance_diff(
    data, "r2", "Model Performance Difference - R-squared (R2)"
)

# Plot 3 - Mean Absolute Error (MAE) Difference vs. Time Difference
plot_and_save_performance_diff(
    data, "mae", "Model Performance Difference - Mean Absolute Error (MAE)"
)

# Plot 4 - Root Mean Squared Error (RMSE) Difference vs. Time Difference
plot_and_save_performance_diff(
    data, "rmse", "Model Performance Difference - Root Mean Squared Error (RMSE)"
)
