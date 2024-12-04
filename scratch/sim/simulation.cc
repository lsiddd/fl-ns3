// Macro for logging
#define LOG(x) std::cout << x << std::endl

// Project-specific headers
#include "MyApp.h"
#include "client_types.h"
#include "notifications.h"
#include "utils.h"
#include "dataframe.h"

// External library headers
#include "json.hpp"

// NS-3 module headers
#include "ns3/applications-module.h"
#include "ns3/command-line.h"
#include "ns3/config-store-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"

// Standard Library headers
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <tuple>

// Using declarations for convenience
using namespace ns3;
// using namespace mmwave;
using json = nlohmann::json;

// Define the simulation logging component
NS_LOG_COMPONENT_DEFINE("Simulation");

// Global constants
static constexpr double simStopTime = 500.0;
static constexpr int numberOfUes = 20;
static constexpr int numberOfEnbs = 1;
static constexpr int numberOfParticipatingClients = numberOfUes;
static constexpr int scenarioSize = 1000;
std::string algorithm = "fedavg";

DataFrame accuracy_df;
DataFrame participation_df;
DataFrame throughput_df;

// Global variables for simulation objects
NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;
NetDeviceContainer enbDevs;
NetDeviceContainer ueDevs;
Ipv4Address remoteHostAddr;

// Random number generation setup
std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(0, scenarioSize);

// Flow monitoring helper
FlowMonitorHelper flowmon;

// Data structure for tracking various metrics
std::map<Ipv4Address, double> endOfStreamTimes;
std::map<Ptr<Node>, int> nodeModelSize, nodeTrainingTime;
std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;

// Global state variables
static bool roundFinished = true;
static int roundNumber = 0;

// Client-related information
std::vector<NodesIps> nodesIPs;
std::vector<ClientModels> clientsInfo;
std::vector<ClientModels> selectedClients;

// Timeout for certain operations
Time timeout = Seconds(120);
static double constexpr managerInterval = 0.1;

void initializeDataFrames() {
    // List of columns for accuracy_df
    std::vector<std::string> accuracy_columns = {
        "time", "user", "round", "accuracy", "compressed_size",
        "compressed_top_n_size", "duration", "loss",
        "number_of_samples", "uncompressed_size", "val_accuracy", "val_loss"
    };

    // List of columns for participation_df
    std::vector<std::string> participation_columns = {
        "time", "participating", "total"
    };

    // List of columns for throughput_df
    std::vector<std::string> throughput_columns = {
        "time", "tx_throughput", "rx_throughput"
    };

    // Initialize columns for accuracy_df
    for (const auto& column : accuracy_columns) {
        accuracy_df.addColumn(column);
    }

    // Initialize columns for participation_df
    for (const auto& column : participation_columns) {
        participation_df.addColumn(column);
    }

    // Initialize columns for throughput_df
    for (const auto& column : throughput_columns) {
        throughput_df.addColumn(column);
    }
}

std::pair<double, double> getRsrpSinr(uint32_t nodeIdx) {
    // Retrieve the UE device for the given node index
    Ptr<NetDevice> ueDevice = ueDevs.Get(nodeIdx);

    // Retrieve the LTE UE net device object and RRC instance
    auto lteUeNetDevice = ueDevice->GetObject<LteUeNetDevice>();
    auto rrc = lteUeNetDevice->GetRrc();

    // Extract the RNTI and Cell ID from the RRC instance
    auto rnti = rrc->GetRnti();
    auto cellId = rrc->GetCellId();

    // Retrieve RSRP and SINR for the given cell ID and RNTI
    double rsrp = rsrpUe[cellId][rnti];
    double sinr = sinrUe[cellId][rnti];

    // Return the RSRP and SINR as a pair
    return {rsrp, sinr};
}

// Helper function to check if a file exists
bool fileExists(const std::string &filename) {
    if (!std::filesystem::exists(filename)) {
        // std::cerr << "Error: File does not exist: " << filename << std::endl;
        return false;
    }
    return true;
}

// Helper function to parse JSON file
json parseJsonFile(const std::string &filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        std::cerr << "Error: Failed to open file: " << filepath << std::endl;
        throw std::runtime_error("File not found or cannot be opened");
    }

    try {
        json j;
        ifs >> j;  // Parse JSON from the file
        return j;
    } catch (const json::parse_error& e) {
        std::cerr << "Error: Failed to parse JSON from file " << filepath 
                  << " - " << e.what() << std::endl;
        throw std::runtime_error("JSON parsing error");
    }
}



int getCompressionFactor() {
    if (algorithm == "fedavg") {
        return 1;
    } else if (algorithm == "fedprox") {
        return 2;
    } else if (algorithm == "weighted_fedavg") {
        return 1.5;
    } else if (algorithm == "pruned_fedavg") {
        return 3;
    }
    return 1;  // Default factor
}

void simulateDummyTraining(std::vector<ClientModels>& clientsInfo) {
    const int nodeTrainingTime = 5000;  // Constant training time of 5 seconds
    const int bytes = 22910076;         // Base uncompressed model size

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        auto [rsrp, sinr] = getRsrpSinr(i);
        double dummyAcc = 0.8;
        clientsInfo.emplace_back(ueNodes.Get(i), nodeTrainingTime, bytes, rsrp, sinr, dummyAcc);
    }
}

void issueTrainingRequests() {
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        std::stringstream cmd;
        cmd << "curl -X POST \"http://127.0.0.1:8182/train\" -H \"Content-Type: application/json\" -d '{"
            << "\"n_clients\": " << ueNodes.GetN() << ", "
            << "\"client_id\": " << i << ", "
            << "\"epochs\": 1, "
            << "\"model\": \"models/" << ueNodes.Get(i) << ".keras\", "
            << "\"top_n\": 3, "
            << "\"algorithm\": \"" << algorithm << "\"}'";

        LOG(cmd.str());  // Log the command
        int ret = system(cmd.str().c_str());  // Execute the command

        if (ret != 0) {
            LOG("Command failed with return code: " << ret);
        }
    }
}


void waitForFile(const std::string& filename) {
    while (!fileExists(filename)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

int getBaseModelSize(uint32_t i) {
    std::stringstream modelFile;
    modelFile << "models/" << ueNodes.Get(i) << ".keras";
    int baseModelSize = getFileSize(modelFile.str());
    return baseModelSize / 2;  // Default to uncompressed size
}

int applyAlgorithmSpecificAdjustments(json& j, int baseModelSize, double fedproxMu, int compressionFactor, double sinr, uint32_t i) {
    int adjustedModelSize = j["compressed_size"];

    if (algorithm == "fedprox") {
        adjustedModelSize = static_cast<int>(baseModelSize / compressionFactor);
    } else if (algorithm == "weighted_fedavg") {
        double client_weight = sinr / 100.0;
        adjustedModelSize = static_cast<int>(baseModelSize / client_weight);
    } else if (algorithm == "pruned_fedavg") {
        adjustedModelSize = static_cast<int>(baseModelSize / compressionFactor);
    }

    return adjustedModelSize;
}

double adjustAccuracyBasedOnAlgorithm(json& j, double fedproxMu, int baseModelSize, int compressionFactor, uint32_t i) {
    double accuracy = j["accuracy"];
    double loss = j["loss"];

    if (algorithm == "fedprox") {
        accuracy -= fedproxMu * loss;
    } else if (algorithm == "pruned_fedavg") {
        accuracy *= 0.9;  // Slight accuracy drop due to pruning
    }

    return accuracy;
}

void logMetrics(json& j, double accuracy, int baseModelSize, uint32_t i) {
    accuracy_df.addRow({
        Simulator::Now().GetSeconds(),
        i,
        roundNumber,
        accuracy,
        float(j["compressed_size"]),
        float(j["compressed_top_n_size"]),
        float(j["duration"]),
        float(j["loss"]),
        float(j["number_of_samples"]),
        float(baseModelSize),
        float(j["val_accuracy"]),
        float(j["val_loss"])
    });
}

void cleanupFinishFile(const std::string& finishFile) {
    std::stringstream rmCommand;
    rmCommand << "rm " << finishFile;
    int ret_rm = system(rmCommand.str().c_str());

    if (ret_rm != 0) {
        LOG("Failed to remove finish file, command returned: " << ret_rm);
    }
}


void collectTrainingMetrics(std::vector<ClientModels>& clientsInfo, double fedproxMu, int compressionFactor) {
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        // Wait for the training completion signal file
        std::stringstream finishFile;
        finishFile << "models/" << ueNodes.Get(i) << ".finish";
        waitForFile(finishFile.str());

        // Retrieve model size and other metrics
        int baseModelSize = getBaseModelSize(i);
        auto [rsrp, sinr] = getRsrpSinr(i);
        json j = parseJsonFile("models/" + std::to_string(ueNodes.Get(i)->GetId()) + ".json");

        // Apply algorithm-specific adjustments
        int adjustedModelSize = applyAlgorithmSpecificAdjustments(j, baseModelSize, fedproxMu, compressionFactor, sinr, i);
        double accuracy = adjustAccuracyBasedOnAlgorithm(j, fedproxMu, baseModelSize, compressionFactor, i);

        LOG("Client " << i << " Model Size (Adjusted): " << adjustedModelSize << " bytes");

        // Store client info in the results vector
        clientsInfo.emplace_back(ueNodes.Get(i), j["duration"], adjustedModelSize, rsrp, sinr, accuracy);

        // Log the metrics to a data frame
        logMetrics(j, accuracy, baseModelSize, i);

        // Clean up .finish file for next round
        cleanupFinishFile(finishFile.str());
    }
}

std::vector<ClientModels> trainClients() {
    std::vector<ClientModels> clientsInfo;
    LOG("=================== " << Simulator::Now().GetSeconds() << " seconds.");

    bool dummy = true;  // Toggle for dummy mode
    // bool dummy = true;  // Toggle for dummy mode

    // Set up algorithm-specific parameters
    int compressionFactor = getCompressionFactor();
    double fedproxMu = 0.1;  // FedProx regularization parameter

    // If in dummy mode, simulate training for quick testing
    if (dummy) {
        simulateDummyTraining(clientsInfo);
        return clientsInfo;
    }

    // Step 1: Issue training requests to clients
    issueTrainingRequests();

    // Step 2: Wait for completion and collect metrics
    collectTrainingMetrics(clientsInfo, fedproxMu, compressionFactor);

    return clientsInfo;
}

void getClientsInfo() {
    // Iterate through each UE node
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        // Get the RSRP and SINR for the current UE node
        auto [rsrp, sinr] = getRsrpSinr(i);

        // Retrieve the UE node device and build the filename for JSON
        Ptr<Node> ueNode = ueNodes.Get(i);
        std::string jsonFilename = "models/" + std::to_string(ueNode->GetId()) + ".json";

        try {
            // Parse the existing JSON file for the UE node
            json j = parseJsonFile(jsonFilename);

            // Update the RSRP and SINR values in the JSON object
            j["rsrp"] = rsrp;
            j["sinr"] = sinr;

            // Log the updated JSON (for debugging or output purposes)
            LOG(j);
        } catch (const std::exception& e) {
            // Log an error if the file doesn't exist or parsing fails
            std::cerr << "Error processing " << jsonFilename << ": " << e.what() << std::endl;
        }
    }
}


// Assuming Clients_Models is a structure or class that stores relevant information about clients
std::vector<ClientModels> clientSelectionSinr(int n, std::vector<ClientModels> clientsInfo) {
    // Define a vector to store pairs of SINR and corresponding client
    std::vector<std::pair<double, ClientModels>> sinrClients;
    json selectedClientsJson;

    for (uint32_t i = 0; i < clientsInfo.size(); i++) {
        // Get the SINR for the client using a hypothetical get_rsrp_sinr function
        auto [rsrp, sinr] = getRsrpSinr(i);

        // Store the SINR and client information as a pair
        sinrClients.push_back({sinr, clientsInfo[i]});
    }

    // Sort clients based on their SINR values in descending order
    std::sort(sinrClients.begin(), sinrClients.end(),
    [](const std::pair<double, ClientModels> &a, const std::pair<double, ClientModels> &b) {
        return a.first > b.first; // Compare SINR values
    });

    for (int i = 0; i < n && (long unsigned int)i < sinrClients.size(); ++i) {
        // clients_info.push_back(sinr_clients[i].second);
        clientsInfo[i].selected = true;
        //  = sinr_clients[i].second;
        // client.selected = true;
        std::stringstream modelFilename;
        modelFilename << "models/" << clientsInfo[i].node << ".keras";
        selectedClientsJson["selected_clients"].push_back(modelFilename.str());
    }

    std::ofstream out("selected_clients.json");
    out << std::setw(4) << selectedClientsJson << std::endl;
    out.close();

    return clientsInfo;
}

std::vector<ClientModels> clientSelectionRandom(int n, std::vector<ClientModels> clientsInfo) {
    std::vector<uint32_t> selected(ueNodes.GetN(), 0);
    std::vector<int> numbers(ueNodes.GetN()); // Inclusive range [0, N]

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        numbers[i] = i;
    }

    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::shuffle(numbers.begin(), numbers.end(), gen);
    numbers.resize(n);
    // Create a JSON object to store selected client models
    json selectedClientsJson;

    for (auto i : numbers) {
        LOG(clientsInfo[i]);
        clientsInfo[i].selected = true;
        // Add the model filename of the selected client to the JSON object
        std::stringstream modelFilename;
        modelFilename << "models/" << ueNodes.Get(i) << ".keras";
        selectedClientsJson["selected_clients"].push_back(modelFilename.str());
    }

    // Save the JSON object to a file
    std::ofstream out("selected_clients.json");
    out << std::setw(4) << selectedClientsJson << std::endl;
    out.close();
    return clientsInfo;
}

std::vector<ClientModels> selectClients(int numClients, const std::vector<ClientModels> &clientsInfo) {
    if (algorithm == "fedavg" || algorithm == "fedprox") {
        return clientSelectionRandom(numClients, clientsInfo);
    }

    else if (algorithm == "weighted_fedavg") {
        // Select clients with better signal quality
        return clientSelectionSinr(numClients, clientsInfo);
    }

    else if (algorithm == "pruned_fedavg") {
        // Select clients with higher accuracy scores
        std::vector<std::pair<double, ClientModels>> accuracyClients;

        for (const auto &client : clientsInfo) {
            accuracyClients.push_back({client.accuracy, client});
        }

        // Sort by accuracy in descending order and select top clients
        std::sort(accuracyClients.begin(), accuracyClients.end(),
        [](const auto &a, const auto &b) {
            return a.first > b.first;
        });

        std::vector<ClientModels> selectedClients;

        for (int i = 0; i < numClients && (long unsigned int)i < accuracyClients.size(); ++i) {
            selectedClients.push_back(accuracyClients[i].second);
        }

        return selectedClients;
    }

    return clientSelectionSinr(numClients, clientsInfo);
}

void logServerEvaluation() {
    // Helper function to parse JSON file with error handling
    auto parseJsonFile = [](const std::string &filepath) -> nlohmann::json {
        std::ifstream ifs(filepath);
        nlohmann::json j;

        // Check if file stream is valid (file exists and is readable)
        if (!ifs.is_open()) {
            LOG("Error: Could not open file " << filepath);
            return {}; // Return an empty JSON object
        }

        try {
            ifs >> j; // Attempt to parse the JSON
        }

        catch (nlohmann::json::parse_error &e) {
            LOG("Error: Failed to parse JSON file " << filepath << ". Error: " << e.what());
            return {}; // Return an empty JSON object on parse failure
        }

        return j; // Return the parsed JSON object if successful
    };
    std::string jsonFile = "evaluation_metrics.json";
    // Call the parse function and handle the result
    nlohmann::json j = parseJsonFile(jsonFile);

    if (j.is_null()) {
        LOG("Error: No valid data found in the JSON file.");
    }

    else {
        LOG(Simulator::Now().GetSeconds() << " seconds, round number  " << roundNumber << " " << j); // Log the JSON content if parsing was successful
    }
}

void aggregation() {
    if (algorithm == "fedavg") {
        runScriptAndMeasureTime("scratch/server.py --aggregation fedavg");
    }

    else if (algorithm == "fedprox") {
        runScriptAndMeasureTime("scratch/server.py --aggregation fedprox");
    }

    else if (algorithm == "weighted_fedavg") {
        runScriptAndMeasureTime("scratch/server.py --aggregation weighted_fedavg");
    }

    else if (algorithm == "pruned_fedavg") {
        runScriptAndMeasureTime("scratch/server.py --aggregation pruned_fedavg");
    }

    logServerEvaluation();
}

// void aggregation()
// {
//     // if (algorithm == "flips")
//     // {
//     //     runScriptAndMeasureTime("scratch/server_flips.py");
//     // }
//     // else
//     // {
//     //     runScriptAndMeasureTime("scratch/server.py");
//     // }
//     if (algorithm == "compressed")
//     {

//         runScriptAndMeasureTime("scratch/server.py --aggregation fedavg");
//     }
//     else if (algorithm == "uncompressed")
//     {
//         runScriptAndMeasureTime("scratch/server.py --aggregation fedprox");
//     }
//     else if (algorithm == "compressed_top_n_size")
//     {
//         runScriptAndMeasureTime("scratch/server.py --agregation weighted_fedavg");
//     }
//     logServerEvaluation();
// }

void sendModelsToServer(const std::vector<ClientModels> &clients) {
    for (const auto &client : clients) {
        if (client.selected) {
            int adjustedSize = client.nodeModelSize;

            if (algorithm == "pruned_fedavg") {
                adjustedSize *= 0.5; // Simulate pruning by reducing data size
            }

            LOG("Client " << client.node << " scheduling send model of size " << adjustedSize << " bytes.");
            Simulator::Schedule(MilliSeconds(client.nodeTrainingTime),
                                &sendStream,
                                client.node,
                                remoteHostContainer.Get(0),
                                adjustedSize);
        }
    }
}

// void sendModelsToServer(std::vector<ClientModels> clients)
// {
//     for (auto i : clients)
//     {
//         if (i.selected)
//         {
//             LOG("Client " << i << " scheduling send model.");
//             Simulator::Schedule(MilliSeconds(i.nodeTrainingTime),
//                                 &sendStream,
//                                 i.node,
//                                 remoteHostContainer.Get(0),
//                                 i.nodeModelSize);
//         }
//     }
// }

void writeSuccessfulClients() {
    json successfulClientsJson;

    for (const auto &client : selectedClients) {
        // Check if the client successfully sent their model
        Ptr<Ipv4> ipv4 = client.node->GetObject<Ipv4>();
        Ipv4Address clientIp = ipv4->GetAddress(1, 0).GetLocal(); // Get the client's IP address

        if (endOfStreamTimes.find(clientIp) != endOfStreamTimes.end()) {
            // If the client has finished sending, log the model in JSON
            std::stringstream modelFilename;
            modelFilename << "models/" << client.node << ".keras";
            successfulClientsJson["successful_clients"].push_back(modelFilename.str());
        }
    }

    // Save the successful clients' models to a JSON file
    std::ofstream out("successful_clients.json");
    out << std::setw(4) << successfulClientsJson << std::endl;
    out.close();
}

void manager() {
    static Time roundStart;

    if (algorithm == "flips" && endOfStreamTimes.size() > numberOfParticipatingClients * 2 / 3) {
        roundFinished = true;
        LOG("Round timed out, not all clients were able to send " << endOfStreamTimes.size() << "/"
            << numberOfParticipatingClients);
        participation_df.addRow({Simulator::Now().GetSeconds(), float(endOfStreamTimes.size()), numberOfParticipatingClients});
    }

    if (Simulator::Now() - roundStart > timeout) {
        roundFinished = true;
        LOG("Round timed out, not all clients were able to send " << endOfStreamTimes.size() << "/"
            << numberOfParticipatingClients);
        participation_df.addRow({Simulator::Now().GetSeconds(), float(endOfStreamTimes.size()), numberOfParticipatingClients});
    }

    nodesIPs = nodeToIps();

    if (roundFinished) {
        if (roundNumber != 0) {
            LOG("Round finished at " << Simulator::Now().GetSeconds()
                << ", all clients were able to send! "
                << endOfStreamTimes.size() << "/" << numberOfParticipatingClients);
            writeSuccessfulClients();
            aggregation();
        }

        participation_df.addRow({Simulator::Now().GetSeconds(), float(endOfStreamTimes.size()), numberOfParticipatingClients});

        roundCleanup();
        roundStart = Simulator::Now();
        roundNumber++;
        roundFinished = false;
        LOG("Starting round " << roundNumber << " at " << Simulator::Now().GetSeconds()
            << " seconds.");
        clientsInfo = trainClients();
        // selected_clients = client_selection(numberOfParticipatingClients, clients_info);
        // selectedClients = clientSelectionSinr(numberOfParticipatingClients, clientsInfo);
        selectedClients = selectClients(numberOfParticipatingClients, clientsInfo);

        sendModelsToServer(selectedClients);

        accuracy_df.toCsv("accuracy.csv");
    }

    participation_df.toCsv("clientParticipation.csv");
    throughput_df.toCsv("throughput.csv");
    roundFinished = checkFinishedTransmission(nodesIPs, selectedClients);
    Simulator::Schedule(Seconds(1), &manager);
}

void ConfigureDefaults() {
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024 * 10));
    Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(MilliSeconds(200)));
    Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout", TimeValue(Seconds(2)));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(131072 * 100 * 10));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(131072 * 100 * 10));
    Config::SetDefault("ns3::ComponentCarrier::PrimaryCarrier", BooleanValue(true));
    Config::SetDefault("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));
    // Config::SetDefault("ns3::LteHelper::UsePdschForCqiGeneration", BooleanValue(true));
    // Config::SetDefault("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue(true));
    // Config::SetDefault("ns3::LteUePowerControl::ClosedLoop", BooleanValue(true));
    Config::SetDefault("ns3::LteUePowerControl::AccumulationEnabled", BooleanValue(false));
    // Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(43.0));
    // lower the ue tx power for more challenging transmission
    Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(20.0));

    // Config::SetDefault("ns3::PhasedArrayModel::AntennaElement",
    //                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    // LogComponentEnable("MmWaveLteRrcProtocolReal", LOG_LEVEL_ALL);
    // LogComponentEnable("mmWaveRrcProtocolIdeal", LOG_LEVEL_ALL);
    // LogComponentEnable("MmWaveUeNetDevice", LOG_LEVEL_ALL);
    // Config::SetDefault("ns3::ComponentCarrier::UlBandwidth", UintegerValue(15));
    // Config::SetDefault("ns3::ComponentCarrier::DlBandwidth", UintegerValue(15));
}

// Main function
int main(int argc, char *argv[]) {
    ConfigureDefaults();

    initializeDataFrames();

    // CommandLine cmd;
    // cmd.Parse(argc, argv);

    CommandLine cmd;
    cmd.AddValue("algorithm", "Select the aggregation method for federated learning", algorithm);
    cmd.Parse(argc, argv);

    // set up helpers
    Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    mmwaveHelper->SetEpcHelper(epcHelper);
    mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    mmwaveHelper->SetHandoverAlgorithmAttribute("ServingCellThreshold", UintegerValue(30));
    mmwaveHelper->SetHandoverAlgorithmAttribute("NeighbourCellOffset", UintegerValue(1));
    // mmwaveHelper->SetAttribute("PathlossModel", StringValue("ns3::Cost231PropagationLossModel"));

    ConfigStore inputConfig;
    inputConfig.ConfigureDefaults();

    // set up remote host
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));

    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    remoteHostAddr = internetIpIfaces.GetAddress(1);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    enbNodes.Create(numberOfEnbs);
    ueNodes.Create(numberOfUes);

    MobilityHelper enbmobility;
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    MobilityHelper uemobility;
    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        uePositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }

    for (uint32_t i = 0; i < enbNodes.GetN(); i++) {
        enbPositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }

    std::string traceFile = "mobility_traces/campus.ns_movements";
    Ns2MobilityHelper ns2 = Ns2MobilityHelper(traceFile);
    ns2.Install(ueNodes.Begin(), ueNodes.End());
    // uemobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    // uemobility.SetPositionAllocator(uePositionAlloc);
    // uemobility.Install(ueNodes);

    enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbmobility.SetPositionAllocator(enbPositionAlloc);
    enbmobility.Install(enbNodes);
    enbmobility.Install(pgw);
    enbmobility.Install(remoteHost);

    enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
    mmwaveHelper->AddX2Interface(enbNodes);
    mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);
    // mmwaveHelper->EnableTraces();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        // Access the LteUePhy from the UE device
        Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();
        // Connect trace source to monitor SINR and RSRP
        uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr",
                                          MakeCallback(&ReportUeSinrRsrp));
    }

    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    Simulator::Schedule(Seconds(managerInterval), &manager);
    Simulator::Schedule(Seconds(managerInterval), &networkInfo, monitor);

    AnimationInterface anim("mmwave-animation.xml");

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        anim.UpdateNodeDescription(ueNodes.Get(i), "UE");
        anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0);
    }

    for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
        anim.UpdateNodeDescription(enbNodes.Get(i), "ENB");
        anim.UpdateNodeColor(enbNodes.Get(i), 0, 255, 0);
    }

    anim.UpdateNodeDescription(remoteHost, "RH");
    anim.UpdateNodeColor(remoteHost, 0, 0, 255);
    anim.UpdateNodeDescription(pgw, "pgw");
    anim.UpdateNodeColor(pgw, 0, 0, 255);

    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkUe));
    Simulator::Stop(Seconds(simStopTime));
    Simulator::Run();
    std::cout << "End of stream times per IP address:" << std::endl;

    for (const auto &entry : endOfStreamTimes) {
        std::cout << "IP Address: " << entry.first
                  << " received the end signal at time: " << entry.second << " seconds."
                  << std::endl;
    }

    Simulator::Destroy();
    return 0;
}
