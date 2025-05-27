# Filename: plot_participation.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the participation CSV file
participation_csv_path = 'clientParticipation_fl_api.csv'

# Check if the file exists
if not os.path.exists(participation_csv_path):
    print(f"Error: Client participation data file not found at '{participation_csv_path}'")
    print("Please ensure you have run the ns-3 simulation and it generated this file.")
else:
    print(f"Reading client participation data from '{participation_csv_path}'...")
    # Read the CSV file into a pandas DataFrame
    try:
        df_participation = pd.read_csv(participation_csv_path)

        print("Client participation data loaded successfully.")
        print("DataFrame head:")
        print(df_participation.head())

        # Check if necessary columns exist
        required_cols = ['time', 'round', 'selected_in_ns3', 'participated_in_ns3_comms']
        if not all(col in df_participation.columns for col in required_cols):
            print(f"Error: Required columns not found in '{participation_csv_path}'.")
            print(f"Required: {required_cols}")
            print(f"Found: {list(df_participation.columns)}")
        else:
            # --- Plotting Number of Clients Selected and Participated ---
            plt.figure(figsize=(10, 6))
            plt.plot(df_participation['time'], df_participation['selected_in_ns3'], marker='o', linestyle='-', label='Selected in ns-3')
            plt.plot(df_participation['time'], df_participation['participated_in_ns3_comms'], marker='x', linestyle='--', color='orange', label='Completed ns-3 Comms')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Number of Clients')
            plt.title('Client Selection and Participation Over Simulation Time')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            participation_count_plot_filename = 'client_participation_count_over_time.png'
            plt.savefig(participation_count_plot_filename)
            print(f"Client participation count plot saved as '{participation_count_plot_filename}'")

            # --- Optional: Plotting Participation Rate ---
            # Avoid division by zero if selected_in_ns3 is 0
            df_participation['participation_rate'] = df_participation.apply(
                lambda row: row['participated_in_ns3_comms'] / row['selected_in_ns3'] if row['selected_in_ns3'] > 0 else 0,
                axis=1
            )
            plt.figure(figsize=(10, 6))
            plt.plot(df_participation['time'], df_participation['participation_rate'], marker='o', linestyle='-', color='green')
            plt.xlabel('Simulation Time (s)')
            plt.ylabel('Participation Rate (Completed / Selected)')
            plt.title('Client Participation Rate Over Simulation Time')
            plt.ylim(0, 1.1) # Rate is between 0 and 1
            plt.grid(True)
            plt.tight_layout()
            participation_rate_plot_filename = 'client_participation_rate_over_time.png'
            plt.savefig(participation_rate_plot_filename)
            print(f"Client participation rate plot saved as '{participation_rate_plot_filename}'")

            # Display plots (optional, uncomment to show interactive plots)
            # plt.show()

    except Exception as e:
        print(f"An error occurred while processing '{participation_csv_path}': {e}")
