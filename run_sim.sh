#!/bin/bash

# Enable safe deletion: stops on errors
set -e

# Function to handle Ctrl+C (SIGINT) and stop both processes
cleanup() {
    echo "Stopping processes..."

    # Graceful kill first
    kill $client_pid $ns3_pid 2>/dev/null
    sleep 2  # Give processes time to terminate gracefully

    # Forceful kill if processes are still running
    kill -9 $client_pid $ns3_pid 2>/dev/null || true

    # Wait for processes to finish
    wait $client_pid $ns3_pid 2>/dev/null
    echo "Processes stopped."
    exit 0
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Suppress the errors about missing files in removal commands
rm -f models/*
rm -f metrics*
rm -f selected*
rm -f *.json
rm -f client_exec.txt
rm -f .txt
rm -f *.png
rm -f metrics.txt
rm -f simulation_output.txt

# Run the Python client in the background and capture its PID
nohup python scratch/client.py > client_exec.txt 2>&1 &
client_pid=$!

sleep 5

# Run the ns3 simulation in the background, showing output with tee
./ns3 run simulation 2>&1 | tee simulation_output.txt &
ns3_pid=$!

# Wait for the ns3 process to finish
wait $ns3_pid

# After ns3 finishes, kill the Python client process
echo "ns3 process finished. Stopping client.py..."
kill $client_pid 2>/dev/null

# If client.py doesn't stop gracefully, force kill
sleep 2
kill -9 $client_pid 2>/dev/null || true

# Wait for client to finish
wait $client_pid 2>/dev/null || true

python plot_results.py

echo "Processes stopped."
