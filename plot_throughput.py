# Filename: plot_throughput.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the throughput CSV file
throughput_csv_path = 'throughput_fl_api.csv'

# Check if the file exists
if not os.path.exists(throughput_csv_path):
    print(f"Error: Throughput data file not found at '{throughput_csv_path}'")
    print("Please ensure you have run the ns-3 simulation and it generated this file.")
else:
    print(f"Reading throughput data from '{throughput_csv_path}'...")
    # Read the CSV file into a pandas DataFrame
    try:
        df_throughput = pd.read_csv(throughput_csv_path)

        print("Throughput data loaded successfully.")
        print("DataFrame head:")
        print(df_throughput.head())

        # Check if necessary columns exist
        required_cols = ['time', 'tx_throughput_mbps', 'rx_throughput_mbps']
        if not all(col in df_throughput.columns for col in required_cols):
            print(f"Error: Required columns not found in '{throughput_csv_path}'.")
            print(f"Required: {required_cols}")
            print(f"Found: {list(df_throughput.columns)}")
        else:
            # --- Plotting TX Throughput ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_throughput['time'], df_throughput['tx_throughput_mbps'], marker='o', linestyle='-', color='blue', label='TX Throughput')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Throughput (Mbps)')
            plt.title('TX Throughput Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            tx_throughput_plot_filename = 'tx_throughput_over_time.png'
            plt.savefig(tx_throughput_plot_filename)
            print(f"TX throughput plot saved as '{tx_throughput_plot_filename}'")

            # --- Plotting RX Throughput ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_throughput['time'], df_throughput['rx_throughput_mbps'], marker='o', linestyle='-', color='green', label='RX Throughput')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Throughput (Mbps)')
            plt.title('RX Throughput Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            rx_throughput_plot_filename = 'rx_throughput_over_time.png'
            plt.savefig(rx_throughput_plot_filename)
            print(f"RX throughput plot saved as '{rx_throughput_plot_filename}'")

            # --- Optional: Combined Throughput Plot ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_throughput['time'], df_throughput['tx_throughput_mbps'], label='TX Throughput')
            plt.plot(df_throughput['time'], df_throughput['rx_throughput_mbps'], label='RX Throughput')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Throughput (Mbps)')
            plt.title('TX and RX Throughput Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            combined_throughput_plot_filename = 'combined_throughput_over_time.png'
            plt.savefig(combined_throughput_plot_filename)
            print(f"Combined throughput plot saved as '{combined_throughput_plot_filename}'")


            # Display plots (optional, uncomment to show interactive plots)
            # plt.show()

    except Exception as e:
        print(f"An error occurred while processing '{throughput_csv_path}': {e}")
