#!/bin/bash

# Enable safe deletion: stops on errors


rm models/*
rm metrics* 
rm selected* 
rm *.json 
rm *.png 
rm *.csv 
rm metrics.txt 

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

# Define different values for numberOfUes and algorithm to iterate over
ues_values=(2 10 20 30 40 50)  # Adjust these values as needed
algorithms=("fedavg" "fedprox" "weighted_fedavg" "pruned_fedavg")

# Main simulation directory to store all runs
mkdir -p simulations

# Loop over each combination of numberOfUes and algorithm
for num_ues in "${ues_values[@]}"; do
    for algorithm in "${algorithms[@]}"; do
        # Define a directory for each parameter combination
        run_dir="simulations/ues_${num_ues}_alg_${algorithm}"
        mkdir -p "$run_dir"

        echo "Running simulation with numberOfUes = $num_ues and algorithm = $algorithm"

        # Update the numberOfUes value in the simulation file
        sed -i "s/numberOfUes =.*$/numberOfUes = $num_ues;/g" scratch/sim/simulation.cc

        # Update the algorithm value in the simulation file
        sed -i "s/algorithm = \".*\";/algorithm = \"$algorithm\";/g" scratch/sim/simulation.cc

        # Run the Python client in the background and capture its PID
        nohup python scratch/client.py | tee "$run_dir/client_exec.txt" 2>&1 &
        client_pid=$!

        sleep 10

        # Run the ns3 simulation in the background, showing output with tee
        ./ns3 run simulation 2>&1 | tee "$run_dir/simulation_output.txt" &
        ns3_pid=$!

        # Wait for the ns3 process to finish
        wait $ns3_pid

        # After ns3 finishes, kill the Python client process
        echo "ns3 process finished for numberOfUes = $num_ues and algorithm = $algorithm. Stopping client.py..."
        kill $client_pid 2>/dev/null

        # If client.py doesn't stop gracefully, force kill
        sleep 2
        kill -9 $client_pid 2>/dev/null || true

        # Wait for client to finish
        wait $client_pid 2>/dev/null || true

        # Move generated files to the directory for this run
        mv metrics* "$run_dir/" 2>/dev/null || true
        mv selected* "$run_dir/" 2>/dev/null || true
        mv *.json "$run_dir/" 2>/dev/null || true
        mv *.png "$run_dir/" 2>/dev/null || true
        mv *.csv "$run_dir/" 2>/dev/null || true
        mv metrics.txt "$run_dir/" 2>/dev/null || true

        echo "Files for numberOfUes = $num_ues and algorithm = $algorithm stored in $run_dir"
    done
done

# After all simulations, generate plots
python plot_results.py
python plot_participation.py
python plot_throughput.py

echo "All simulations completed. Processes stopped."
