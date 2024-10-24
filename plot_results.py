import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    """
    Parse the log file and extract timestamps, RX throughput, client accuracy, validation accuracy,
    and server-side accuracy and participation per round.
    
    :param filename: Path to the log file
    :return: Parsed data including timestamps, RX throughput, client accuracies, validation accuracies,
             and server validation accuracies and client participation over time.
    """
    timestamps = []
    rx_throughput = []
    accuracy_data = {}
    server_validation_acc = []
    client_participation = []
    round_markers = []
    
    # Regular expressions for parsing different log entries
    rx_throughput_pattern = re.compile(r'(\d+(?:\.\d+)?)s: Instant Network Throughput: (\d+(?:\.\d+)?) Mbps')
    client_accuracy_pattern = re.compile(r'Client (\d+) info.+?"(accuracy)":(\d\.\d+),.*?"(val_accuracy)":(\d.\d+)')
    round_end_pattern = re.compile(r'(\d+)\sseconds, round number\s+(\d+)\s+.*"Validation Accuracy":(\d+\.\d+).*"Validation Loss":(\d+\.\d+)')
    round_start_pattern = re.compile(r'Starting round (\d+) at (\d+) seconds')

    with open(filename, 'r') as file:
        for line in file:
            # Match RX throughput data
            match_throughput = rx_throughput_pattern.search(line)
            if match_throughput:
                timestamp = float(match_throughput.group(1))
                throughput = float(match_throughput.group(2))
                timestamps.append(timestamp)
                rx_throughput.append(throughput)
            
            # Match client accuracy data
            match_accuracy = client_accuracy_pattern.search(line)
            if match_accuracy:
                client_id = int(match_accuracy.group(1))
                accuracy = float(match_accuracy.group(3))
                val_accuracy = float(match_accuracy.group(5))
                
                if client_id not in accuracy_data:
                    accuracy_data[client_id] = {'time': [], 'accuracy': [], 'val_accuracy': []}
                
                current_time = timestamps[-1] if timestamps else 0
                accuracy_data[client_id]['time'].append(current_time)
                accuracy_data[client_id]['accuracy'].append(accuracy)
                accuracy_data[client_id]['val_accuracy'].append(val_accuracy)

            # Match server validation accuracy and participation
            match_round_end = round_end_pattern.search(line)
            if match_round_end:
                timestamp = float(match_round_end.group(1))
                round_num = int(match_round_end.group(2))
                val_acc = float(match_round_end.group(3))
                server_validation_acc.append((timestamp, val_acc))
                round_markers.append((timestamp, 'end'))

            # Mark the start of a round
            match_round_start = round_start_pattern.search(line)
            if match_round_start:
                timestamp = float(match_round_start.group(2))
                round_markers.append((timestamp, 'start'))

        file.seek(0)
        lines = file.readlines()
        # print("".join(lines))
        round_participation_pattern = re.compile(r'able to send (\d)\/(\d)\n.*\n.*\n(\d+) seconds, round number  (\d)')
        # match_participation = 
        # print(match_participation)
        # if match_participation:
        for m in round_participation_pattern.finditer("".join(lines)):
            print(m)
            if m:
                timestamp = float(m.group(3))
                clients_participated = int(m.group(1))
                total_clients = int(m.group(2))
                print(timestamp, clients_participated)
                participation_ratio = clients_participated / total_clients
                client_participation.append((timestamp, participation_ratio))


    
    return timestamps, rx_throughput, accuracy_data, server_validation_acc, client_participation, round_markers

def plot_throughput(timestamps, rx_throughput, output_filename):
    """
    Plot the network RX throughput over time and save the plot.
    
    :param timestamps: List of timestamps
    :param rx_throughput: List of RX throughput values
    :param output_filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, rx_throughput, label='Network RX Throughput')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RX Throughput (Mbps)')
    plt.title('Network RX Throughput Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()

def plot_client_accuracy(accuracy_data, output_filename):
    """
    Plot the accuracy for each client over time and save the plot.
    
    :param accuracy_data: Dictionary containing client accuracy and val_accuracy over time
    :param output_filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    for client_id, data in accuracy_data.items():
        plt.plot(data['time'], data['accuracy'], label=f'Client {client_id} Accuracy', marker='o')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Client Accuracy Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()

def plot_average_accuracy(accuracy_data, output_filename):
    """
    Plot the average accuracy and average validation accuracy over time and save the plot.
    
    :param accuracy_data: Dictionary containing client accuracy and val_accuracy over time
    :param output_filename: Filename to save the plot
    """
    all_times = sorted({time for data in accuracy_data.values() for time in data['time']})
    
    avg_accuracy = []
    avg_val_accuracy = []

    for time_point in all_times:
        accuracies = []
        val_accuracies = []
        
        for data in accuracy_data.values():
            if time_point in data['time']:
                idx = data['time'].index(time_point)
                accuracies.append(data['accuracy'][idx])
                val_accuracies.append(data['val_accuracy'][idx])

        avg_accuracy.append(np.mean(accuracies) if accuracies else None)
        avg_val_accuracy.append(np.mean(val_accuracies) if val_accuracies else None)

    # Plot average accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(all_times, avg_accuracy, label='Average Accuracy', marker='o')
    plt.plot(all_times, avg_val_accuracy, label='Average Validation Accuracy', marker='o')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy and Validation Accuracy Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()

def plot_server_accuracy(server_validation_acc, client_participation, round_markers, output_filename):
    """
    Plot the server validation accuracy and client participation over time, with round markers.
    
    :param server_validation_acc: List of tuples (timestamp, validation accuracy)
    :param client_participation: List of tuples (timestamp, participation ratio)
    :param round_markers: List of tuples (timestamp, 'start' or 'end')
    :param output_filename: Filename to save the plot
    """
    timestamps, val_accs = zip(*server_validation_acc)
    timestamps_part, participation_ratios = zip(*client_participation)

    plt.figure(figsize=(10, 6))

    # Plot validation accuracy
    plt.plot(timestamps, val_accs, label='Validation Accuracy (Server)', marker='o')

    # Plot client participation
    plt.plot(timestamps_part, participation_ratios, label='Client Participation', marker='x')

    # Add round markers
    for timestamp, marker_type in round_markers:
        if marker_type == 'start':
            plt.axvline(x=timestamp, color='green', linestyle='--', label='Round Start' if timestamp == round_markers[0][0] else "")
        elif marker_type == 'end':
            plt.axvline(x=timestamp, color='red', linestyle='--', label='Round End' if timestamp == round_markers[0][0] else "")

    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy / Participation')
    plt.title('Server Validation Accuracy and Client Participation Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_throughput.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Parse the log file
    timestamps, rx_throughput, accuracy_data, server_validation_acc, client_participation, round_markers = parse_log_file(log_file)
    
    # Check if any RX throughput data was found
    if not timestamps or not rx_throughput:
        print("No RX throughput data found in the log file.")
        sys.exit(1)
    
    # Plot the throughput data and save the plot
    plot_throughput(timestamps, rx_throughput, 'network_rx_throughput.png')
    
    # Plot client accuracy over time and save the plot
    if accuracy_data:
        plot_client_accuracy(accuracy_data, 'client_accuracy.png')
        plot_average_accuracy(accuracy_data, 'average_accuracy.png')
    else:
        print("No client accuracy data found in the log file.")
    
    # Plot server validation accuracy and client participation over time with round markers
    if server_validation_acc and client_participation:
        plot_server_accuracy(server_validation_acc, client_participation, round_markers, 'server_validation_accuracy.png')
    else:
        print("No server accuracy or client participation data found in the log file.")
