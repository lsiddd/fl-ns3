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
using json = nlohmann::json;

// Define the simulation logging component
NS_LOG_COMPONENT_DEFINE("Simulation");

// Global constants
static constexpr double simStopTime = 1200.0;
static constexpr int numberOfUes = 20;
static constexpr int numberOfEnbs = 5;
static constexpr int numberOfParticipatingClients = numberOfUes;
static constexpr int scenarioSize = 1000;
std::string algorithm = "weighted_fedavg";

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
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        try {
            // Retrieve RSRP and SINR for the UE node
            auto [rsrp, sinr] = getRsrpSinr(i);

            // Build JSON filename using the UE node's ID
            const auto ueNodeId = ueNodes.Get(i)->GetId();
            const std::string jsonFilename = "models/" + std::to_string(ueNodeId) + ".json";

            // Parse existing JSON file and update values
            auto jsonContent = parseJsonFile(jsonFilename);
            jsonContent["rsrp"] = rsrp;
            jsonContent["sinr"] = sinr;

            // Log updated JSON for debugging or output
            LOG(jsonContent);
        } catch (const std::exception& ex) {
            // Log errors during file processing
            std::cerr << "Error processing UE node " << i << ": " << ex.what() << std::endl;
        }
    }
}

std::vector<ClientModels> clientSelectionSinr(int n, std::vector<ClientModels> clientsInfo)
{
    // Define a vector to store pairs of SINR and corresponding client
    std::vector<std::pair<double, ClientModels>> sinrClients;
    json selectedClientsJson;

    for (uint32_t i = 0; i < clientsInfo.size(); i++)
    {
        // Get the SINR for the client using a hypothetical get_rsrp_sinr function
        auto [rsrp, sinr] = getRsrpSinr(i);

        // Store the SINR and client information as a pair
        sinrClients.push_back({sinr, clientsInfo[i]});
    }

    // Sort clients based on their SINR values in descending order
    std::sort(sinrClients.begin(), sinrClients.end(),
              [](const std::pair<double, ClientModels> &a, const std::pair<double, ClientModels> &b)
              {
                  return a.first > b.first; // Compare SINR values
              });

    for (int i = 0; i < n && (long unsigned int)i < sinrClients.size(); ++i)
    {
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

void selectClientsByAccuracy(int numClients, const std::vector<ClientModels>& clientsInfo, std::vector<ClientModels>& selectedClients) {
    // Create a vector of pairs of accuracy and corresponding client
    std::vector<std::pair<double, ClientModels>> accuracyClients;
    for (const auto& client : clientsInfo) {
        accuracyClients.emplace_back(client.accuracy, client);
    }

    // Sort by accuracy in descending order
    std::sort(accuracyClients.begin(), accuracyClients.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Select top 'numClients' clients based on accuracy
    for (int i = 0; i < numClients && i < static_cast<int>(accuracyClients.size()); ++i) {
        selectedClients.push_back(accuracyClients[i].second);
    }
}

std::vector<ClientModels> selectClients(int numClients, const std::vector<ClientModels>& clientsInfo) {
    // Handle algorithm selection using a more concise approach
    if (algorithm == "fedavg" || algorithm == "fedprox") {
        return clientSelectionRandom(numClients, clientsInfo);
    }
    if (algorithm == "weighted_fedavg") {
        // Select clients based on SINR for better signal quality
        return clientSelectionSinr(numClients, clientsInfo);
    }
    if (algorithm == "pruned_fedavg") {
        // Select clients based on accuracy (higher accuracy preferred)
        std::vector<ClientModels> selectedClients;
        selectClientsByAccuracy(numClients, clientsInfo, selectedClients);
        return selectedClients;
    }

    // Default case if the algorithm is not recognized
    return clientSelectionSinr(numClients, clientsInfo);
}


void logServerEvaluation() {
    // Helper function to parse JSON file with error handling
    auto parseJsonFile = [](const std::string& filepath) -> nlohmann::json {
        std::ifstream ifs(filepath);
        if (!ifs.is_open()) {
            LOG("Error: Could not open file " << filepath);
            return {};  // Return an empty JSON object if file can't be opened
        }

        try {
            nlohmann::json j;
            ifs >> j;  // Attempt to parse the JSON
            return j;
        } catch (const nlohmann::json::parse_error& e) {
            LOG("Error: Failed to parse JSON file " << filepath << ". Error: " << e.what());
            return {};  // Return an empty JSON object on parse failure
        }
    };

    // Define the path to the JSON file
    const std::string jsonFile = "evaluation_metrics.json";

    // Parse the JSON file
    nlohmann::json j = parseJsonFile(jsonFile);

    if (j.is_null()) {
        // Log an error if JSON parsing failed or the file was empty
        LOG("Error: No valid data found in the JSON file.");
    } else {
        // Log the content of the JSON along with the current simulator time and round number
        LOG(Simulator::Now().GetSeconds() << " seconds, round number " << roundNumber << " " << j.dump(4));
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

void sendModelsToServer(const std::vector<ClientModels>& clients) {
    for (const auto& client : clients) {
        if (client.selected) {
            // Adjust model size based on algorithm (e.g., pruned models)
            double adjustedSize = static_cast<double>(client.nodeModelSize);  // Ensure size is treated as a double
            if (algorithm == "pruned_fedavg") {
                adjustedSize *= 0.5;  // Simulate pruning by reducing data size
            }

            // Log the scheduling of model send
            LOG("Client " << client.node->GetId() << " scheduling send model of size " << adjustedSize << " bytes.");

            // Schedule the model send after the client training time
            Simulator::Schedule(MilliSeconds(client.nodeTrainingTime), &sendStream,
                                client.node, remoteHostContainer.Get(0), adjustedSize);
        }
    }
}
void writeSuccessfulClients() {
    json successfulClientsJson;

    for (const auto& client : selectedClients) {
        // Get the client's IP address
        Ptr<Ipv4> ipv4 = client.node->GetObject<Ipv4>();
        Ipv4Address clientIp = ipv4->GetAddress(1, 0).GetLocal();  // Client's IP address

        // Check if the client has finished sending their model
        if (endOfStreamTimes.find(clientIp) != endOfStreamTimes.end()) {
            // Construct the model filename and add it to the successful clients list
            successfulClientsJson["successful_clients"].push_back(
                "models/" + std::to_string(client.node->GetId()) + ".keras"
            );
        }
    }

    // Write the successful clients' models to a JSON file
    std::ofstream outFile("successful_clients.json");
    if (outFile.is_open()) {
        outFile << std::setw(4) << successfulClientsJson << std::endl;
    } else {
        std::cerr << "Error: Failed to open successful_clients.json for writing." << std::endl;
    }
}


bool isRoundTimedOut(Time &roundStart) {
    if (algorithm == "flips" && endOfStreamTimes.size() > numberOfParticipatingClients * 2 / 3) {
        return true;
    }

    return Simulator::Now() - roundStart > timeout;
}

void logRoundTimeout() {
    LOG("Round timed out, not all clients were able to send " << endOfStreamTimes.size() << "/"
        << numberOfParticipatingClients);
}

void addParticipationToDataFrame() {
    participation_df.addRow({Simulator::Now().GetSeconds(), float(endOfStreamTimes.size()), numberOfParticipatingClients});
}

void finalizeRound() {
    if (roundNumber != 0) {
        LOG("Round finished at " << Simulator::Now().GetSeconds()
            << ", all clients were able to send! "
            << endOfStreamTimes.size() << "/" << numberOfParticipatingClients);
        writeSuccessfulClients();
        aggregation();
    }

    addParticipationToDataFrame();
    roundCleanup();
}

void startNewRound(Time roundStart) {
    roundStart = Simulator::Now();
    roundNumber++;
    roundFinished = false;

    LOG("Starting round " << roundNumber << " at " << Simulator::Now().GetSeconds() << " seconds.");

    // Prepare the next round's clients
    clientsInfo = trainClients();
    selectedClients = selectClients(numberOfParticipatingClients, clientsInfo);

    sendModelsToServer(selectedClients);
}

void exportDataFrames() {
    accuracy_df.toCsv("accuracy.csv");
    participation_df.toCsv("clientParticipation.csv");
    throughput_df.toCsv("throughput.csv");
}

void manager() {
    static Time roundStart;

    // Check if the round is finished based on algorithm and timeout
    if (isRoundTimedOut(roundStart)) {
        roundFinished = true;
        logRoundTimeout();
        addParticipationToDataFrame();
    }


    nodesIPs = nodeToIps();

    if (roundFinished) {
        finalizeRound();
        startNewRound(roundStart);
    }

    // Export participation, throughput, and accuracy data
    exportDataFrames();

    // Check if transmission is finished
    roundFinished = checkFinishedTransmission(nodesIPs, selectedClients);

    // Schedule the next manager call
    Simulator::Schedule(Seconds(1), &manager);
}

void ConfigureDefaults() {
    // LTE RLC (Radio Link Control) Configuration
    const uint32_t maxTxBufferSizeUm = 10 * 1024 * 1024 * 10;
    const uint32_t maxTxBufferSizeAm = 10 * 1024 * 1024;
    const uint32_t maxTxBufferSizeLowLat = 10 * 1024 * 1024;

    // Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(maxTxBufferSizeUm));
    // Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue(maxTxBufferSizeAm));
    // Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue(maxTxBufferSizeLowLat));

    // TCP Configuration
    const uint32_t sndRcvBufSize = 131072 * 100 * 10;

    Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(MilliSeconds(200)));
    Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout", TimeValue(Seconds(2)));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(sndRcvBufSize));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(sndRcvBufSize));

    // LTE PHY (Physical Layer) and Spectrum Configuration
    Config::SetDefault("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue(true));

    // LTE Helper Configuration
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));
    Config::SetDefault("ns3::LteUePowerControl::AccumulationEnabled", BooleanValue(false));

    // LTE UE (User Equipment) Power Configuration
    const double txPowerUe = 20.0;  // Lower the UE transmission power for more challenging transmission
    Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(txPowerUe));

    // LTE Component Carrier Configuration
    Config::SetDefault("ns3::ComponentCarrier::PrimaryCarrier", BooleanValue(true));

    // Debugging and Logging Configuration (commented out, can be enabled if needed)
    // LogComponentEnable("MmWaveLteRrcProtocolReal", LOG_LEVEL_ALL);
    // LogComponentEnable("mmWaveRrcProtocolIdeal", LOG_LEVEL_ALL);
    // LogComponentEnable("MmWaveUeNetDevice", LOG_LEVEL_ALL);

    // Placeholder for additional configurations
    // Config::SetDefault("ns3::ComponentCarrier::UlBandwidth", UintegerValue(15));
    // Config::SetDefault("ns3::ComponentCarrier::DlBandwidth", UintegerValue(15));
}


// Constants for configuration values
const std::string ALGORITHM_DESC = "Select the aggregation method for federated learning";
const DataRate NETWORK_DATA_RATE = DataRate("100Gb/s");
const uint32_t MTU_SIZE = 1500;
const Time P2P_CHANNEL_DELAY = MicroSeconds(1);
const double UE_TX_POWER = 20.0;
const Time SIM_STOP_TIME = Seconds(simStopTime);
const Time MANAGER_INTERVAL = Seconds(managerInterval);

// Main function
int main(int argc, char *argv[]) {
    // Configure defaults for the simulation
    ConfigureDefaults();

    // Initialize data frames
    initializeDataFrames();

    // Parse command line arguments
    CommandLine cmd;
    cmd.AddValue("algorithm", ALGORITHM_DESC, algorithm);
    cmd.Parse(argc, argv);

    // Create LTE helper and EPC helper
    Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    mmwaveHelper->SetEpcHelper(epcHelper);
    mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    mmwaveHelper->SetHandoverAlgorithmAttribute("ServingCellThreshold", UintegerValue(30));
    mmwaveHelper->SetHandoverAlgorithmAttribute("NeighbourCellOffset", UintegerValue(1));

    // Configure the simulation environment
    ConfigStore inputConfig;
    inputConfig.ConfigureDefaults();

    // Set up remote host and network devices
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(NETWORK_DATA_RATE));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(MTU_SIZE));
    p2ph.SetChannelAttribute("Delay", TimeValue(P2P_CHANNEL_DELAY));

    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    remoteHostAddr = internetIpIfaces.GetAddress(1);

    // Static routing setup for remote host
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // Create nodes for eNB and UE
    enbNodes.Create(numberOfEnbs);
    ueNodes.Create(numberOfUes);

    // Set up mobility models
    MobilityHelper enbmobility, uemobility;
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        uePositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }
    for (uint32_t i = 0; i < enbNodes.GetN(); i++) {
        enbPositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }

    // Install mobility models
    std::string traceFile = "mobility_traces/campus.ns_movements";
    Ns2MobilityHelper ns2(traceFile);
    ns2.Install(ueNodes.Begin(), ueNodes.End());

    enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbmobility.SetPositionAllocator(enbPositionAlloc);
    enbmobility.Install(enbNodes);
    enbmobility.Install(pgw);
    enbmobility.Install(remoteHost);

    // Install devices and attach them
    enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
    mmwaveHelper->AddX2Interface(enbNodes);
    mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);

    // Set up static routing for UEs
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // Monitor SINR and RSRP for UEs
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();
        uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr", MakeCallback(&ReportUeSinrRsrp));
    }

    // Flow monitoring
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Schedule events
    Simulator::Schedule(MANAGER_INTERVAL, &manager);
    Simulator::Schedule(MANAGER_INTERVAL, &networkInfo, monitor);

    // Set up animation for visualization
    AnimationInterface anim("mmwave-animation.xml");
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        anim.UpdateNodeDescription(ueNodes.Get(i), "UE");
        anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0);  // Red color for UE
    }
    for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
        anim.UpdateNodeDescription(enbNodes.Get(i), "ENB");
        anim.UpdateNodeColor(enbNodes.Get(i), 0, 255, 0);  // Green color for eNB
    }
    anim.UpdateNodeDescription(remoteHost, "RH");
    anim.UpdateNodeColor(remoteHost, 0, 0, 255);  // Blue color for Remote Host
    anim.UpdateNodeDescription(pgw, "PGW");
    anim.UpdateNodeColor(pgw, 0, 0, 255);  // Blue color for PGW

    // Connect simulation events
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

    // Run the simulation
    Simulator::Stop(SIM_STOP_TIME);
    Simulator::Run();

    // Output end of stream times
    std::cout << "End of stream times per IP address:" << std::endl;
    for (const auto &entry : endOfStreamTimes) {
        std::cout << "IP Address: " << entry.first
                  << " received the end signal at time: " << entry.second << " seconds."
                  << std::endl;
    }

    // Cleanup and destroy simulation
    Simulator::Destroy();
    return 0;
}
