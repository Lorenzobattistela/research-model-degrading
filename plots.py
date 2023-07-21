import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from the CSV file
data = pd.read_csv("polynomial_regression_results.csv")

# Calculate the time difference between training and prediction years
data["time_difference"] = data["year-predicted"] - data["year-trained"]

# Create scatter plots for performance differences
fig, axes = plt.subplots(2, 2, figsize=(12, 8))


# Function to plot performance difference vs. time difference
def plot_performance_diff(data, metric, ax):
    sns.scatterplot(data=data, x="time_difference", y=metric, ax=ax)
    ax.set_xlabel("Time Difference (Years)")
    ax.set_ylabel(f"{metric} Difference")
    ax.axhline(y=0, color="red", linestyle="--")  # Reference line at y=0 (no change)


# Plot 1 - Mean Squared Error (MSE) Difference vs. Time Difference
plot_performance_diff(data, "mse", ax=axes[0, 0])

# Plot 2 - R-squared (R2) Difference vs. Time Difference
plot_performance_diff(data, "r2", ax=axes[0, 1])

# Plot 3 - Mean Absolute Error (MAE) Difference vs. Time Difference
plot_performance_diff(data, "mae", ax=axes[1, 0])

# Plot 4 - Root Mean Squared Error (RMSE) Difference vs. Time Difference
plot_performance_diff(data, "rmse", ax=axes[1, 1])

# Add titles to the subplots
axes[0, 0].set_title("Model Performance Difference - Mean Squared Error (MSE)")
axes[0, 1].set_title("Model Performance Difference - R-squared (R2)")
axes[1, 0].set_title("Model Performance Difference - Mean Absolute Error (MAE)")
axes[1, 1].set_title("Model Performance Difference - Root Mean Squared Error (RMSE)")

# Adjust layout to prevent overlapping of titles and labels
plt.tight_layout()

# Show the plots
plt.show()
