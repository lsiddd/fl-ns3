# Federated Learning Simulation using NS-3 and LTE/MmWave

## Overview

This project simulates a federated learning environment over a mobile network using the NS-3 simulator. The simulation involves several key elements such as mobile user devices (UEs), eNodeBs (base stations), and a remote server. Federated learning clients (UEs) train local models and periodically send their updates to the server over a 4G LTE or 5G MmWave network.

The simulation is designed to handle parallel client training, transmission of trained models, and server aggregation in a federated learning round.

### Key Components:
- **UE Nodes**: User equipment (e.g., mobile devices) participating in the federated learning process.
- **eNB Nodes**: The base stations that handle communication between UEs and the server.
- **Remote Host**: The server that receives the locally trained models and performs aggregation.

## Requirements

1. **NS-3**: This simulation uses NS-3 and its LTE/MmWave modules.
2. **Python 3**: For running the local training tasks via scripts.
3. **NetAnim**: Optionally used for visualizing the network simulation.
4. **Dependencies**: Make sure NS-3 modules like `lte`, `internet`, `mobility`, and `applications` are installed and available.

### Key Libraries/Modules:
- `ns3/applications-module.h`
- `ns3/internet-module.h`
- `ns3/lte-helper.h`
- `ns3/flow-monitor-module.h`
- `json.hpp`: For handling JSON files.
- `random`, `future`: For generating randomness and handling parallel execution.

## Compilation

1. Clone this repository.
2. Compile the NS-3 simulation:
    ```sh
    ./ns3 configure
    ./ns3
    ```
3. Run the simulation:
    ```sh
    ./ns3 --run "simulation"
    ```

## Features

- **Federated Learning Process**: 
  - Each UE trains a local model using a Python script.
  - Models are sent over the network to the remote server.
  - The server aggregates the models.

- **LTE/MmWave Network Simulation**: 
  - The UEs communicate with the server via base stations using LTE or MmWave networks.
  - Mobility and handovers are supported between eNBs.

- **Parallel Execution**: 
  - Client training is run in parallel using C++ futures for efficiency.

- **Network Monitoring**: 
  - Instantaneous throughput and connection details are logged during the simulation.

## Code Structure

### Classes

- **MyApp**: Simulates application layer traffic generation for UEs. This class handles the sending of data packets from UEs to the server.
  
- **Clients_Models**: Represents each federated learning client, including its associated node, training time, and bytes transferred.

### Key Functions

- **sendstream()**: Sends the trained model from the UE to the server.
- **train_clients()**: Triggers the training process for all clients, each running a Python script to train a model.
- **manager()**: Manages the federated learning rounds, selects clients, and coordinates the sending of models to the server.

### Event Callbacks

- **NotifyConnectionEstablishedUe/NotifyConnectionEstablishedEnb**: Logs the connection establishment between UEs and eNBs.
- **NotifyHandoverStartUe/NotifyHandoverEndOkUe**: Logs UE handover events during the simulation.
  
### Simulation Setup

The simulation involves:
1. **Network Configuration**: 
   - UEs and eNBs are initialized with random positions.
   - LTE or MmWave channels are set up.
2. **Mobility Models**: UEs are assigned a constant position, but this can be changed to a dynamic model if required.
3. **Data Transfer**: Each UE sends its trained model to the server, and throughput is monitored using the `FlowMonitor` class.

## Running the Simulation

1. **Configure the number of UEs and eNBs**: These are set via global variables:
    ```cpp
    int number_of_ues = 10;
    int number_of_enbs = 1;
    int n_participaping_clients = 5;
    ```

2. **Training and Model Transmission**: The UEs perform local training, and the trained model is sent to the server over the network. The server then aggregates the models.
   
3. **Timeout Mechanism**: If a round of transmission does not finish in a specified time (`timeout = Seconds(60)`), the round is considered timed out.

4. **Output**: The simulation logs network events and the time taken by each UE to send its model.

## Output

- **Logs**: Simulation progress and network events are logged to the console.
- **Throughput Measurement**: Instantaneous throughput in Mbps is calculated and displayed.
- **Client Selection**: Each round, a subset of clients is randomly selected to participate in federated learning.

## License

This project is licensed under the MIT License.

