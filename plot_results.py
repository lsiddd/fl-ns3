import pandas as pd
import matplotlib.pyplot as plt
import os


# Configure matplotlib para usar o backend 'Agg'
plt.switch_backend('Agg')

# Load the dataset
data = pd.read_csv("accuracy.csv")

# Create a directory to save plots if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Calculate the duration of each federated learning round by time difference
# data['round_duration'] = data.groupby('round')['time'].transform(lambda x: x.max() - x.min())
# Calculate the duration between rounds by finding time differences across rounds
round_times = data.groupby('round')['time'].min().reset_index()
round_times['round_duration'] = round_times['time'].diff().fillna(0)
data = data.merge(round_times[['round', 'round_duration']], on='round', how='left')

# Define individual functions for each metric plot type and save the plot
def plot_accuracy(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.plot(user_data['round'], user_data['accuracy'], label=f'User {user}')
    plt.title("Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/accuracy_over_rounds.png")
    plt.close()

def plot_val_accuracy(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.plot(user_data['round'], user_data['val_accuracy'], label=f'User {user}', marker='o')
    plt.title("Validation Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Validation Accuracy")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/val_accuracy_over_rounds.png")
    plt.close()

def plot_loss(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.plot(user_data['round'], user_data['loss'], label=f'User {user}', marker='o')
    plt.title("Loss over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/loss_over_rounds.png")
    plt.close()

def plot_val_loss(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.plot(user_data['round'], user_data['val_loss'], label=f'User {user}', marker='o')
    plt.title("Validation Loss over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Validation Loss")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/val_loss_over_rounds.png")
    plt.close()

def plot_compressed_size(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.bar(user_data['round'], user_data['compressed_size'], label=f'User {user}')
    plt.title("Compressed Size over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Compressed Size (bytes)")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/compressed_size_over_rounds.png")
    plt.close()

def plot_compressed_top_n_size(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.bar(user_data['round'], user_data['compressed_top_n_size'], label=f'User {user}')
    plt.title("Compressed Top-N Size over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Compressed Top-N Size (bytes)")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/compressed_top_n_size_over_rounds.png")
    plt.close()

def plot_duration(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.plot(user_data['round'], user_data['duration'], label=f'User {user}', marker='o')
    plt.title("Duration per User over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Duration (s)")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/duration_per_user_over_rounds.png")
    plt.close()

def plot_number_of_samples(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.bar(user_data['round'], user_data['number_of_samples'], label=f'User {user}')
    plt.title("Number of Samples per User over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Number of Samples")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/number_of_samples_over_rounds.png")
    plt.close()

def plot_uncompressed_size(data):
    plt.figure(figsize=(10, 6))
    for user in data['user'].unique():
        user_data = data[data['user'] == user]
        plt.bar(user_data['round'], user_data['uncompressed_size'], label=f'User {user}')
    plt.title("Uncompressed Size over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Uncompressed Size (bytes)")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/uncompressed_size_over_rounds.png")
    plt.close()

def plot_round_duration(data):
    plt.figure(figsize=(10, 6))
    rounds = data['round'].unique()
    round_durations = data.groupby('round')['round_duration'].mean()
    plt.plot(rounds, round_durations, marker='o')
    plt.title("Average Duration of Each Federated Learning Round")
    plt.xlabel("Round")
    plt.ylabel("Duration (s)")
    plt.savefig(f"{output_dir}/round_duration_over_rounds.png")
    plt.close()
    

# Define functions to plot metrics with averages and standard deviation
def plot_metric_with_std(data, metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    grouped_data = data.groupby('round')[metric].agg(['mean', 'std']).reset_index()
    plt.errorbar(grouped_data['round'], grouped_data['mean'], yerr=grouped_data['std'], fmt='-o', capsize=5)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.savefig(f"{output_dir}/{metric}_average_with_std.png")
    plt.close()

# Plot metrics with averages and standard deviation
plot_metric_with_std(data, 'accuracy', "Average Accuracy over Rounds", "Accuracy")
plot_metric_with_std(data, 'val_accuracy', "Average Validation Accuracy over Rounds", "Validation Accuracy")
plot_metric_with_std(data, 'loss', "Average Loss over Rounds", "Loss")
plot_metric_with_std(data, 'val_loss', "Average Validation Loss over Rounds", "Validation Loss")
plot_metric_with_std(data, 'compressed_size', "Average Compressed Size over Rounds", "Compressed Size (bytes)")
plot_metric_with_std(data, 'compressed_top_n_size', "Average Compressed Top-N Size over Rounds", "Compressed Top-N Size (bytes)")
plot_metric_with_std(data, 'uncompressed_size', "Average Uncompressed Size over Rounds", "Uncompressed Size (bytes)")

def plot_duration_with_std(data):
    plt.figure(figsize=(10, 6))
    grouped_duration = data.groupby('round')['round_duration'].agg(['mean', 'std']).reset_index()
    plt.errorbar(grouped_duration['round'], grouped_duration['mean'], yerr=grouped_duration['std'], fmt='-o', capsize=5)
    plt.title("Average Duration of Each Federated Learning Round")
    plt.xlabel("Round")
    plt.ylabel("Duration (s)")
    plt.savefig(f"{output_dir}/round_duration_average_with_std.png")
    plt.close()

# Call the function to plot the duration with standard deviation
plot_duration_with_std(data)

# Call each plot function to save all plots
plot_accuracy(data)
plot_val_accuracy(data)
plot_loss(data)
plot_val_loss(data)
plot_compressed_size(data)
plot_compressed_top_n_size(data)
plot_duration(data)
plot_number_of_samples(data)
plot_uncompressed_size(data)
plot_round_duration(data)

print(f"Plots saved in the '{output_dir}' directory.")
