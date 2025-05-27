# Filename: plot_accuracy.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the accuracy CSV file
accuracy_csv_path = 'accuracy_fl_api.csv'

# Check if the file exists
if not os.path.exists(accuracy_csv_path):
    print(f"Error: Accuracy data file not found at '{accuracy_csv_path}'")
    print("Please ensure you have run the ns-3 simulation and it generated this file.")
else:
    print(f"Reading accuracy data from '{accuracy_csv_path}'...")
    # Read the CSV file into a pandas DataFrame
    try:
        df_accuracy = pd.read_csv(accuracy_csv_path)

        print("Accuracy data loaded successfully.")
        print("DataFrame head:")
        print(df_accuracy.head())

        # Check if necessary columns exist
        required_cols = ['time', 'round', 'global_accuracy', 'global_loss', 'avg_client_accuracy', 'avg_client_loss']
        if not all(col in df_accuracy.columns for col in required_cols):
            print(f"Error: Required columns not found in '{accuracy_csv_path}'.")
            print(f"Required: {required_cols}")
            print(f"Found: {list(df_accuracy.columns)}")
        else:
            # --- Plotting Global Model Accuracy ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_accuracy['time'], df_accuracy['global_accuracy'], marker='o', linestyle='-', label='Global Test Accuracy')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Accuracy')
            plt.title('Global Model Accuracy Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            accuracy_plot_filename = 'global_accuracy_over_time.png'
            plt.savefig(accuracy_plot_filename)
            print(f"Global accuracy plot saved as '{accuracy_plot_filename}'")

            # --- Plotting Global Model Loss ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_accuracy['time'], df_accuracy['global_loss'], marker='o', linestyle='-', color='red', label='Global Test Loss')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Loss')
            plt.title('Global Model Loss Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            loss_plot_filename = 'global_loss_over_time.png'
            plt.savefig(loss_plot_filename)
            print(f"Global loss plot saved as '{loss_plot_filename}'")

            # --- Optional: Plotting Average Client Metrics ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_accuracy['time'], df_accuracy['avg_client_accuracy'], marker='x', linestyle='--', color='green', label='Average Client Accuracy (Weighted)')
            plt.plot(df_accuracy['time'], df_accuracy['avg_client_loss'], marker='+', linestyle='--', color='purple', label='Average Client Loss (Weighted)')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Metric Value')
            plt.title('Average Client Metrics Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            avg_client_metrics_filename = 'avg_client_metrics_over_time.png'
            plt.savefig(avg_client_metrics_filename)
            print(f"Average client metrics plot saved as '{avg_client_metrics_filename}'")


            # Display plots (optional, uncomment to show interactive plots)
            # plt.show()

    except Exception as e:
        print(f"An error occurred while processing '{accuracy_csv_path}': {e}")
