import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure matplotlib for non-GUI backend
plt.switch_backend('Agg')

# Load the dataset
data = pd.read_csv("throughput.csv")

# Create a directory to save plots if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Plot TX Throughput over Time
def plot_tx_throughput(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['tx_throughput'], label="TX Throughput", color="blue")
    plt.title("TX Throughput Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("TX Throughput (Mbps)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/tx_throughput_over_time.png")
    plt.close()

# Plot RX Throughput over Time
def plot_rx_throughput(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['rx_throughput'], label="RX Throughput", color="green")
    plt.title("RX Throughput Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("RX Throughput (Mbps)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/rx_throughput_over_time.png")
    plt.close()

# Plot TX and RX Throughput Together
def plot_combined_throughput(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['time'], data['tx_throughput'], label="TX Throughput", color="blue")
    plt.plot(data['time'], data['rx_throughput'], label="RX Throughput", color="green")
    plt.title("Network Throughput Over Time (TX and RX)")
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/combined_throughput_over_time.png")
    plt.close()

# Call the plotting functions
plot_tx_throughput(data)
plot_rx_throughput(data)
plot_combined_throughput(data)

print(f"Plots saved in the '{output_dir}' directory.")
