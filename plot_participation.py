import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure matplotlib for non-GUI backend
plt.switch_backend('Agg')

# Load the dataset
data = pd.read_csv("clientParticipation.csv")

# Create a directory to save plots if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Detect new rounds based on changes in time
data['new_round'] = data['time'].diff().fillna(0) != 0

# Plot participation over rounds
def plot_participation_over_rounds(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['participating'], label="Participating Users", marker='o')
    plt.xlabel("Round")
    plt.ylabel("Number of Participating Users")
    plt.title("User Participation Over Rounds")
    plt.grid(True)
    plt.savefig(f"{output_dir}/participation_over_rounds.png")
    plt.close()

# Plot participation over time, marking new rounds
def plot_participation_over_time(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['participating'], label="Participating Users", marker='o')
    plt.xlabel("Time")
    plt.ylabel("Number of Participating Users")
    plt.title("User Participation Over Time")
    
    # Mark the start of each new round
    new_rounds = data[data['new_round']]
    plt.scatter(new_rounds['time'], new_rounds['participating'], color='red', label="Round Start", zorder=5)
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f"{output_dir}/participation_over_time.png")
    plt.close()

# Call plotting functions
plot_participation_over_rounds(data)
plot_participation_over_time(data)

print(f"Plots saved in the '{output_dir}' directory.")
