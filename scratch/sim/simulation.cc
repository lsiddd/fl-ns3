// Filename: /home/lucas/fl-ns3/scratch/sim/simulation.cc
// Macro for logging - REPLACED WITH NS_LOG_COMPONENT_DEFINE
// #define LOG(x) std::cout << x << std::endl

// Project-specific headers
#include "MyApp.h"
#include "client_types.h"
#include "dataframe.h"
#include "notifications.h"
#include "utils.h"

// External library headers
#include "json.hpp"

// NS-3 module headers
#include "ns3/applications-module.h"
#include "ns3/command-line.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h" // Required for Simulator::Schedule
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/lte-ue-rrc.h" // Make sure LteUeRrc is included
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/log.h" // Include NS-3 log module
#include "ns3/tcp-socket.h" // Include TCP Socket headers for logging
#include "ns3/tcp-socket-base.h" // Include TCP Socket Base headers for logging
#include "ns3/ipv4-l3-protocol.h" // Include IPv4 headers for logging

// Standard Library headers
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h> // For sleep
#include <vector>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath> // For std::ceil

// Using declarations for convenience
using namespace ns3;
using json = nlohmann::json;

// Define the simulation logging component
NS_LOG_COMPONENT_DEFINE("Simulation");

// Global constants
static constexpr double simStopTime = 400.0;
static constexpr int numberOfUes = 20; // Reduced for faster testing
static constexpr int numberOfEnbs = 5; // Reduced for faster testing
static constexpr int numberOfParticipatingClients = 15;
static constexpr int scenarioSize = 1000;
bool useStaticClients = true;
std::string algorithm = "fedavg"; // This will map to FL API's aggregation if
                                  // needed, but /run_round uses its own.

// Python FL API settings
const std::string FL_API_BASE_URL = "http://127.0.0.1:5000";
const int FL_API_NUM_CLIENTS = numberOfUes; // Tell FL API about the total UEs
const int FL_API_CLIENTS_PER_ROUND =
    numberOfParticipatingClients; // How many ns-3 selects to tell FL API

DataFrame accuracy_df;
DataFrame participation_df;
DataFrame throughput_df;
DataFrame rsrp_sinr_df; // New DataFrame for RSRP/SINR

// Global variables for simulation objects
NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;
NetDeviceContainer enbDevs;
NetDeviceContainer ueDevs;
Ipv4Address remoteHostAddr;

// Random number generation setup - Not directly used for PositionAllocators below
// std::random_device dev;
// std::mt19937 rng(dev());
// std::uniform_int_distribution<std::mt19937::result_type> dist(0, scenarioSize);

// Flow monitoring helper
FlowMonitorHelper flowmon;

// Data structure for tracking various metrics
std::map<Ipv4Address, double> endOfStreamTimes;
// std::map<Ptr<Node>, int> nodeModelSize, nodeTrainingTime; // These will be
// populated differently
std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;

// Global state variables
static bool roundFinished = true;
static int roundNumber = 0;
static bool fl_api_initialized = false;

// Client-related information
std::vector<NodesIps> nodesIPs;
std::vector<ClientModels>
    clientsInfoGlobal; // Holds info for all potential clients
std::vector<ClientModels>
    selectedClientsForCurrentRound; // Clients selected by ns-3 for current
                                    // round

// Timeout for certain operations
Time timeout = Seconds(50); // ns-3 round timeout for model transfers
static double constexpr managerInterval =
    1.0; // ns-3 manager check interval, increased for clearer logging

// --- Helper function to call Python API ---
// Returns the HTTP status code and optionally saves response to a file
int callPythonApi(const std::string &endpoint,
                  const std::string &method = "POST",
                  const std::string &data = "",
                  const std::string &output_file = "") {
  std::stringstream command;
  command << "curl --max-time 20 --connect-timeout 5 -s -o "; // Max 20s total,
                                                              // 5s to connect
  if (output_file.empty()) {
    command << "/dev/null";
  } else {
    command << output_file;
  }
  command << " -w \"%{http_code}\""; // Output only HTTP status code to stdout

  if (!method.empty()) {
    command << " -X " << method;
  }
  if (method == "POST" && !data.empty()) {
    command << " -H \"Content-Type: application/json\" -d '" << data << "'";
  }
  command << " " << FL_API_BASE_URL << endpoint;

  NS_LOG_INFO("callPythonApi: PREPARING to execute CURL for endpoint "
      << endpoint << " at " << Simulator::Now().GetSeconds()
      << "s. Command: " << command.str());
  if (method == "POST" && !data.empty()) {
      NS_LOG_DEBUG("callPythonApi: Payload: " << data.substr(0, 200) << (data.length() > 200 ? "..." : ""));
  }

  std::fflush(stdout); // Force flush output before potentially blocking popen

  char buffer[128];
  std::string result_str = "";
  FILE *pipe = popen(command.str().c_str(), "r");
  if (!pipe) {
    NS_LOG_ERROR("callPythonApi: popen() FAILED for command: "
        << command.str() << " at " << Simulator::Now().GetSeconds() << "s.");
    std::fflush(stdout);
    return -1;
  }
  NS_LOG_DEBUG("callPythonApi: popen successful, READING from pipe for "
      << endpoint << " at " << Simulator::Now().GetSeconds() << "s.");
  std::fflush(stdout);
  try {
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result_str += buffer;
    }
  }

  catch (const std::exception &e) {
    NS_LOG_ERROR("callPythonApi: Exception while reading from pipe for "
        << endpoint << ": " << e.what() << " at "
        << Simulator::Now().GetSeconds() << "s.");
    std::fflush(stdout);
    pclose(pipe);
    return -2;
  } catch (...) {
    NS_LOG_ERROR("callPythonApi: Unknown exception while reading from pipe for "
        << endpoint << " at " << Simulator::Now().GetSeconds() << "s.");
    std::fflush(stdout);
    pclose(pipe);
    return -3;
  }

  NS_LOG_DEBUG("callPythonApi: FINISHED reading from pipe for "
      << endpoint << ". Raw result_str: '" << result_str << "'"
      << " at " << Simulator::Now().GetSeconds() << "s.");
  std::fflush(stdout);

  int status = pclose(pipe);
  NS_LOG_INFO("callPythonApi: pclose status for "
      << endpoint << ": " << status << " at " << Simulator::Now().GetSeconds()
      << "s. Curl result: " << result_str); // Log result_str here
  std::fflush(stdout);

  if (status == -1) {
    NS_LOG_ERROR("callPythonApi: pclose failed or command not found. Status: " << status);
    return -1;
  } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
    NS_LOG_ERROR("callPythonApi: Curl command exited with status "
        << WEXITSTATUS(status) << ". HTTP code: " << result_str);
    // result_str might contain the http code even if curl itself had an error
    // code (e.g. connection refused)
  }

  try {
    return std::stoi(result_str);
  } catch (const std::invalid_argument &ia) {
    NS_LOG_ERROR("callPythonApi: Invalid argument for stoi: '" << result_str << "'");
    return -1; // Or some other error code
  } catch (const std::out_of_range &oor) {
    NS_LOG_ERROR("callPythonApi: Out of range for stoi: '" << result_str << "'");
    return -1; // Or some other error code
  }
}

void initializeDataFrames() {
  std::vector<std::string> accuracy_columns = {"time",
                                               "round",
                                               "global_accuracy",
                                               "global_loss",
                                               "avg_client_accuracy",
                                               "avg_client_loss",
                                               "api_round_duration"};
  std::vector<std::string> participation_columns = {
      "time", "round", "selected_in_ns3", "participated_in_ns3_comms"};
  std::vector<std::string> throughput_columns = {"time", "tx_throughput_mbps",
                                                 "rx_throughput_mbps", "total_tx_bytes", "total_rx_bytes"};
  std::vector<std::string> rsrp_sinr_columns = {
      "time", "round", "ue_node_id", "enb_cell_id", "ue_rnti", "rsrp_dbm", "sinr_db", "connected_state"
  };


  for (const auto &column : accuracy_columns) {
    accuracy_df.addColumn(column);
    NS_LOG_DEBUG("Added accuracy_df column: " << column);
  }
  for (const auto &column : participation_columns) {
    participation_df.addColumn(column);
    NS_LOG_DEBUG("Added participation_df column: " << column);
  }
  for (const auto &column : throughput_columns) {
    throughput_df.addColumn(column);
    NS_LOG_DEBUG("Added throughput_df column: " << column);
  }
  for (const auto &column : rsrp_sinr_columns) {
    rsrp_sinr_df.addColumn(column);
    NS_LOG_DEBUG("Added rsrp_sinr_df column: " << column);
  }
  NS_LOG_INFO("All DataFrames initialized with columns.");
}

std::pair<double, double> getRsrpSinr(uint32_t nodeIdx) {
  Ptr<NetDevice> ueDevice = ueDevs.Get(nodeIdx);
  if (!ueDevice) {
    NS_LOG_DEBUG("getRsrpSinr: UE device at index " << nodeIdx << " is null.");
    return {0.0, 0.0};
  }
  auto lteUeNetDevice = ueDevice->GetObject<LteUeNetDevice>();
  if (!lteUeNetDevice) {
    NS_LOG_DEBUG("getRsrpSinr: NetDevice at index " << nodeIdx << " is not an LteUeNetDevice.");
    return {0.0, 0.0};
  }
  auto rrc = lteUeNetDevice->GetRrc();
  // if (!rrc || !rrc->IsConnected()) return {0.0, 0.0}; // Check if RRC is
  // valid and connected

  std::string connected_state = "NOT_CONNECTED";
  if (!rrc || (rrc->GetState() != LteUeRrc::CONNECTED_NORMALLY &&
               rrc->GetState() != LteUeRrc::CONNECTED_HANDOVER)) {
    // Alternative check: if (!rrc || rrc->GetRnti() ==
    // LteUeRrc::UNINITIALIZED_RNTI)
    NS_LOG_DEBUG("getRsrpSinr: UE Node " << ueNodes.Get(nodeIdx)->GetId() << " RRC not in connected state. State: " << (rrc ? rrc->GetState() : LteUeRrc::IDLE_START));
    rsrp_sinr_df.addRow({Simulator::Now().GetSeconds(), roundNumber, ueNodes.Get(nodeIdx)->GetId(), (uint32_t)0, (uint32_t)0, 0.0, 0.0, connected_state});
    return {0.0, 0.0}; // Not connected or RRC not available
  }
  
  connected_state = (rrc->GetState() == LteUeRrc::CONNECTED_NORMALLY ? "CONNECTED_NORMALLY" : "CONNECTED_HANDOVER");
  auto rnti = rrc->GetRnti();
  auto cellId = rrc->GetCellId();

  double rsrp = 0.0;
  double sinr = 0.0;
  if (rsrpUe.count(cellId) && rsrpUe[cellId].count(rnti)) {
    rsrp = rsrpUe[cellId][rnti];
  }
  if (sinrUe.count(cellId) && sinrUe[cellId].count(rnti)) {
    sinr = sinrUe[cellId][rnti];
  }
  
  // Log RSRP/SINR to DataFrame
  rsrp_sinr_df.addRow({Simulator::Now().GetSeconds(), roundNumber, ueNodes.Get(nodeIdx)->GetId(), (uint32_t)cellId, (uint32_t)rnti, rsrp, sinr, connected_state});
  NS_LOG_DEBUG("getRsrpSinr: UE Node " << ueNodes.Get(nodeIdx)->GetId() << " (CellId: " << cellId << ", RNTI: " << rnti << ") RSRP: " << rsrp << " dBm, SINR: " << sinr << " dB. State: " << connected_state);
  return {rsrp, sinr};
}

// Fills clientsInfoGlobal with current data for ALL UEs
void updateAllClientsGlobalInfo() {
  NS_LOG_INFO("Updating global client information for all UEs.");
  clientsInfoGlobal.clear();
  const int defaultTrainingTime =
      5000; // ms, time ns-3 client "prepares" before sending
  const int defaultModelSizeBytes =
      2000000; // 2MB, placeholder for client model update size

  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    auto [rsrp, sinr] = getRsrpSinr(i);
    // Accuracy here is a placeholder or could be from a previous global round
    // Python API will handle actual accuracies.
    double placeholderAccuracy = 0.1;
    clientsInfoGlobal.emplace_back(ueNodes.Get(i), defaultTrainingTime,
                                   defaultModelSizeBytes, rsrp, sinr,
                                   placeholderAccuracy);
    NS_LOG_DEBUG("  UE Node " << ueNodes.Get(i)->GetId() << ": RSRP=" << rsrp << " dBm, SINR=" << sinr << " dB.");
  }
  NS_LOG_INFO("Global client information updated for " << clientsInfoGlobal.size() << " UEs.");
}

// Selects clients based on ns-3 criteria for the current round
// Populates `selectedClientsForCurrentRound`
void selectNs3ManagedClients(int n_to_select) {
  NS_LOG_INFO("Selecting " << n_to_select << " clients for FL round " << roundNumber << " based on ns-3 criteria.");
  selectedClientsForCurrentRound.clear();
  updateAllClientsGlobalInfo(); // Ensure clientsInfoGlobal is up-to-date with
                                // SINR/RSRP

  // Example: Select top N by SINR (can be random, or your other existing
  // FedAvg/FedProx selections) For simplicity, let's use a SINR-based selection
  // similar to your 'clientSelectionSinr' but operating on clientsInfoGlobal
  // and populating selectedClientsForCurrentRound.

  std::vector<ClientModels> candidates = clientsInfoGlobal; // Copy
  std::sort(candidates.begin(), candidates.end(),
            [](const ClientModels &a, const ClientModels &b) {
              return a.sinr > b.sinr; // Higher SINR is better
            });

  int actual_selected_count = 0;
  for (int i = 0; i < n_to_select && (long unsigned int)i < candidates.size();
       ++i) {
    // Only select clients that are actually connected (SINR != 0.0 indicates a connection in this context)
    if (candidates[i].sinr > 0.001 || candidates[i].rsrp < 0.0) { // Assuming 0.0 means not measured/connected, or very low RSRP
      ClientModels selected_client = candidates[i];
      selected_client.selected =
          true; // Mark as selected for ns-3 comms simulation
      selectedClientsForCurrentRound.push_back(selected_client);
      actual_selected_count++;
      NS_LOG_DEBUG("  Selected client " << selected_client.node->GetId() << " (SINR: " << selected_client.sinr << " dB, RSRP: " << selected_client.rsrp << " dBm)");
    } else {
        NS_LOG_DEBUG("  Skipping client " << candidates[i].node->GetId() << " due to low SINR (" << candidates[i].sinr << " dB) or RSRP (" << candidates[i].rsrp << " dBm).");
    }
  }
  NS_LOG_INFO("ns-3 selected " << actual_selected_count << " clients (out of " << n_to_select << " requested) for FL round " << roundNumber);
  if (actual_selected_count == 0 && n_to_select > 0) {
      NS_LOG_WARN("No eligible clients were selected by ns-3 for this round, possibly due to poor network conditions for all UEs.");
  }
}

// This function is now responsible for:
// 1. Telling Python API which clients ns-3 has selected.
// 2. Python API runs the full FL round (training + aggregation).
// 3. Logging results from Python API.
// 4. Preparing for ns-3 to simulate model uploads from these selected clients.
bool triggerAndProcessFLRoundInApi() {
  NS_LOG_INFO("=================== Triggering FL Round "
      << roundNumber << " in Python API at " << Simulator::Now().GetSeconds()
      << "s ===================");
  std::fflush(stdout); // Ensure log is flushed

  if (selectedClientsForCurrentRound.empty() &&
      FL_API_CLIENTS_PER_ROUND >
          0) // Check if ns-3 wants to select but couldn't
  {
    NS_LOG_INFO("No clients were selected by ns-3 for this round. Skipping API call "
        "for /run_round.");
    std::fflush(stdout);
    if (FL_API_CLIENTS_PER_ROUND > 0)
      return false; // No clients to send to API for training
  }

  // TEST CALL FOR ROUND 2 (Moved for clarity, this logic is specific to test/debug)
  if (roundNumber == 2) {
    NS_LOG_DEBUG("Attempting a TEST CURL to /ping before FL API round 2 /run_round call "
        "at "
        << Simulator::Now().GetSeconds() << "s");
    std::fflush(stdout);
    int test_http_code = callPythonApi("/ping", "GET", "", "ping_response.txt");
    NS_LOG_DEBUG("TEST /ping call HTTP code: " << test_http_code << " at "
                                      << Simulator::Now().GetSeconds() << "s");
    std::fflush(stdout);
    if (test_http_code != 200) {
      NS_LOG_ERROR("ERROR: Test /ping call FAILED. Aborting before /run_round for round "
          "2.");
      std::fflush(stdout);
      return false; // Indicate failure
    }
  }

  json client_indices_payload;
  std::vector<int> indices_list;
  // Only add client.node->GetId() which is uint32_t. The Python API expects
  // int.
  for (const auto &client : selectedClientsForCurrentRound) {
    indices_list.push_back(static_cast<int>(client.node->GetId()));
  }
  client_indices_payload["client_indices"] = indices_list;

  std::string response_file = "fl_round_response.json";
  NS_LOG_INFO("Calling /run_round for round "
      << roundNumber << " with payload (first 100 chars): " << client_indices_payload.dump().substr(0,100) << "...");
  std::fflush(stdout);
  int http_code = callPythonApi("/run_round", "POST",
                                client_indices_payload.dump(), response_file);

  if (http_code == 200) {
    NS_LOG_INFO("Python API /run_round call successful for round " << roundNumber);
    std::ifstream ifs(response_file);
    if (ifs.is_open()) {
      json response_json;
      try {
        ifs >> response_json;
        NS_LOG_INFO("Python API Response (first 200 chars): " << response_json.dump(2).substr(0,200) << "...");
        // Log metrics to accuracy_df
        accuracy_df.addRow(
            {Simulator::Now().GetSeconds(), roundNumber,
             response_json.value("global_test_accuracy", 0.0),
             response_json.value("global_test_loss", 0.0),
             response_json.value("avg_client_accuracy", 0.0),
             response_json.value("avg_client_loss", 0.0),
             response_json.value("round_duration_seconds", 0.0)
            });
        NS_LOG_INFO("Accuracy data added to DataFrame for round " << roundNumber);

        // Update selected clients info with simulated values from API
        auto client_perf_details = response_json.value("simulated_client_performance", json::object());
        NS_LOG_INFO("Updating selected client info with simulated values from API response (" << client_perf_details.size() << " clients)...");
        for (auto const& [client_id_str, perf_data] : client_perf_details.items()) {
            try {
                int client_id = std::stoi(client_id_str);
                // Find the corresponding client in selectedClientsForCurrentRound
                for (auto& client_model_info : selectedClientsForCurrentRound) {
                    if (client_model_info.node->GetId() == (uint32_t)client_id) {
                        client_model_info.nodeTrainingTime = perf_data.value("simulated_training_time_ms", client_model_info.nodeTrainingTime);
                        client_model_info.nodeModelSize = perf_data.value("simulated_model_size_bytes", client_model_info.nodeModelSize);
                        NS_LOG_DEBUG("  Updated client " << client_id << ": training_time=" << client_model_info.nodeTrainingTime << "ms, model_size=" << client_model_info.nodeModelSize << " bytes.");
                        break; // Found and updated
                    }
                }
            } catch (const std::invalid_argument& ia) { NS_LOG_ERROR("  Failed to parse client ID string: " << client_id_str); }
        }


      } catch (json::parse_error &e) {
        NS_LOG_ERROR("ERROR: Failed to parse Python API response JSON from '" << response_file << "': " << e.what());
        return false;
      }
    } else {
      NS_LOG_ERROR("ERROR: Could not open response file: " << response_file);
      return false;
    }
    return true;
  } else {
    NS_LOG_ERROR("ERROR: Python API /run_round call failed. HTTP Code: " << http_code);
    std::ifstream ifs(response_file);
    if (ifs.is_open()) {
      json error_json;
      try {
        ifs >> error_json;
        NS_LOG_ERROR("Python API Error Response: " << error_json.dump(2));
      } catch (json::parse_error &e) {
        NS_LOG_ERROR("ERROR: Failed to parse Python API error JSON from '" << response_file << "': " << e.what());
      }
    }
    return false;
  }
}

void sendModelsToServer() { // Uses selectedClientsForCurrentRound
  NS_LOG_INFO("ns-3: Simulating model uploads for " << selectedClientsForCurrentRound.size() << " selected clients.");
  if (selectedClientsForCurrentRound.empty()) {
      NS_LOG_INFO("  No clients selected for model upload in ns-3 this round. Skipping send.");
      return;
  }

  for (const auto &client_model_info : selectedClientsForCurrentRound) {
    // .selected flag is already true from selectNs3ManagedClients
    // nodeModelSize and nodeTrainingTime are from clientsInfoGlobal (default
    // values) - NOW UPDATED BY API RESPONSE
    NS_LOG_INFO("  Client " << client_model_info.node->GetId()
                  << " scheduling ns-3 send model of size "
                  << client_model_info.nodeModelSize << " bytes "
                  << "after " << client_model_info.nodeTrainingTime
                  << "ms pseudo-training time.");

    Simulator::Schedule(MilliSeconds(client_model_info.nodeTrainingTime),
                        &sendStream, client_model_info.node,
                        remoteHostContainer.Get(0),
                        client_model_info.nodeModelSize);
  }
}

bool isRoundTimedOut(Time roundStartTimeNs3Comms) {
  // Timeout for ns-3 communication phase
  bool timedOut = Simulator::Now() - roundStartTimeNs3Comms > timeout;
  if (timedOut) {
      NS_LOG_WARN("isRoundTimedOut: ns-3 Comms phase for round " << roundNumber << " has timed out at " << Simulator::Now().GetSeconds() << "s.");
  } else {
      NS_LOG_DEBUG("isRoundTimedOut: ns-3 Comms phase for round " << roundNumber << " is not yet timed out. Current duration: " << (Simulator::Now() - roundStartTimeNs3Comms).GetSeconds() << "s.");
  }
  return timedOut;
}

void logRoundTimeout() {
  NS_LOG_WARN("ns-3 Comms Round timed out for round " << roundNumber << ". Successful transfers: "
      << endOfStreamTimes.size() << "/"
      << selectedClientsForCurrentRound.size() << " clients.");
}

void addParticipationToDataFrame() {
  NS_LOG_INFO("Adding participation data to DataFrame for round " << roundNumber << ".");
  participation_df.addRow({Simulator::Now().GetSeconds(), roundNumber,
                           (uint32_t)selectedClientsForCurrentRound.size(),
                           (uint32_t)endOfStreamTimes.size()});
  NS_LOG_INFO("  Recorded selected: " << selectedClientsForCurrentRound.size() << ", completed ns-3 comms: " << endOfStreamTimes.size());
}

// This finalize is for the ns-3 communication part of the round
void finalizeNs3CommsPhase() {
  NS_LOG_INFO("ns-3 Comms phase for round "
      << roundNumber << " finished at " << Simulator::Now().GetSeconds()
      << ". Successful transfers: " << endOfStreamTimes.size() << "/"
      << selectedClientsForCurrentRound.size());

  addParticipationToDataFrame();
  roundCleanup(); // Clears ns-3 apps and endOfStreamTimes
  NS_LOG_INFO("ns-3 Comms phase cleanup complete for round " << roundNumber << ".");
}

void startNewFLRound(
    Time &roundStartTimeNs3CommsParam) // Parameter renamed to avoid conflict
                                       // with static
{
  roundNumber++;
  NS_LOG_INFO("StartNewFLRound: Beginning for FL Round "
      << roundNumber << " at " << Simulator::Now().GetSeconds() << "s.");

  selectNs3ManagedClients(FL_API_CLIENTS_PER_ROUND);
  NS_LOG_INFO("StartNewFLRound: ns-3 selected " << selectedClientsForCurrentRound.size()
                                        << " clients for this round.");

  if (selectedClientsForCurrentRound.empty() && FL_API_CLIENTS_PER_ROUND > 0) {
    NS_LOG_INFO("StartNewFLRound: No clients were selected by ns-3 (e.g., due to SINR "
        "or other criteria, or no eligible UEs). Skipping API call and ns-3 "
        "comms for this round.");
    roundFinished = true; // Mark as finished to allow manager to proceed
    return;
  }

  NS_LOG_INFO("StartNewFLRound: Triggering FL round in Python API for round "
      << roundNumber);
  bool api_success = triggerAndProcessFLRoundInApi(); // This calls Python API
  // triggerAndProcessFLRoundInApi now UPDATES selectedClientsForCurrentRound
  // with actual sim_training_time and sim_model_size from the API response

  if (api_success) {
    NS_LOG_INFO("StartNewFLRound: Python API call successful for round "
        << roundNumber);
    if (!selectedClientsForCurrentRound.empty()) {
      NS_LOG_INFO("StartNewFLRound: Scheduling ns-3 model uploads for "
          << selectedClientsForCurrentRound.size() << " clients with updated times/sizes.");
      sendModelsToServer(); // Schedules ns-3 MyApp instances using the updated info
      roundStartTimeNs3CommsParam =
          Simulator::Now();  // Mark start of ns-3 communication phase
      roundFinished = false; // ns-3 communication phase now active
      NS_LOG_INFO("StartNewFLRound: ns-3 comms phase started for round "
          << roundNumber << " at " << roundStartTimeNs3CommsParam.GetSeconds() << "s. roundFinished set to false.");
    } else {
      NS_LOG_INFO("StartNewFLRound: Python API call successful, but no clients "
          "selected by ns-3 for simulated upload. Marking round as "
          "(comms-wise) finished.");
      roundFinished = true; // No ns-3 comms to simulate
    }
  } else {
    NS_LOG_ERROR("StartNewFLRound: Python API call FAILED for round "
        << roundNumber << ". Skipping ns-3 comms phase.");
    roundFinished = true; // Mark as finished to allow manager to try next FL
                          // round attempt (or stop if max rounds)
  }
}

void exportDataFrames() {
  NS_LOG_INFO("Exporting DataFrames to CSV files.");
  accuracy_df.toCsv("accuracy_fl_api.csv");
  participation_df.toCsv("clientParticipation_fl_api.csv");
  throughput_df.toCsv("throughput_fl_api.csv");
  rsrp_sinr_df.toCsv("rsrp_sinr_metrics.csv"); // Export new RSRP/SINR DataFrame
  NS_LOG_INFO("All DataFrames exported.");
}

void manager() {
  static Time roundStartTimeNs3Comms =
      Simulator::Now(); // Initialize to current time at first call
  NS_LOG_INFO("Manager called at " << Simulator::Now().GetSeconds()
                           << "s. RoundNumber: " << roundNumber
                           << ", roundFinished (ns-3 comms): "
                           << (roundFinished ? "true" : "false"));

  if (!fl_api_initialized) {
    NS_LOG_INFO("Manager: FL API not yet initialized by main. Manager waiting for 5 seconds.");
    Simulator::Schedule(Seconds(5.0), &manager);
    return;
  }

  if (!roundFinished) {
    NS_LOG_INFO("Manager: ns-3 communication phase for round " << roundNumber
                                                       << " is ongoing.");
    if (isRoundTimedOut(roundStartTimeNs3Comms)) {
      NS_LOG_WARN("Manager: Round " << roundNumber << " ns-3 comms timed out.");
      logRoundTimeout(); // Logs successful transfers vs selected
      roundFinished = true;
    } else {
      nodesIPs = nodeToIps(); // Usually static, but good practice
      bool all_selected_clients_finished_ns3_comms =
          checkFinishedTransmission(nodesIPs, selectedClientsForCurrentRound);

      if (all_selected_clients_finished_ns3_comms) {
        if (!selectedClientsForCurrentRound.empty() ||
            !endOfStreamTimes
                 .empty()) { // Avoid logging if no comms were even scheduled
          NS_LOG_INFO("Manager: All selected clients ("
              << endOfStreamTimes.size() << "/"
              << selectedClientsForCurrentRound.size()
              << ") completed ns-3 transmissions for round " << roundNumber);
        } else if (selectedClientsForCurrentRound.empty()) {
          NS_LOG_INFO("Manager: No clients were selected for ns-3 comms in round "
              << roundNumber << ", considering comms phase complete.");
        }
        roundFinished = true;
      } else {
        NS_LOG_INFO("Manager: Waiting for "
            << selectedClientsForCurrentRound.size() - endOfStreamTimes.size()
            << " more clients to finish ns-3 comms for round " << roundNumber);
      }
    }

    if (roundFinished) {
      NS_LOG_INFO("Manager: Finalizing ns-3 comms phase for round " << roundNumber);
      finalizeNs3CommsPhase(); // Logs, adds to participation_df, cleans up apps
    }
  }

  if (roundFinished) {
    NS_LOG_INFO("Manager: ns-3 communication phase for round "
        << roundNumber << " is finished or was skipped.");
    if (roundNumber < 5) // Limit total FL rounds for testing
    {
      NS_LOG_INFO("Manager: Attempting to start new FL round (will be round "
          << roundNumber + 1 << ")");
      startNewFLRound(roundStartTimeNs3Comms); // This will attempt to set
                                               // roundFinished=false
    } else {
      NS_LOG_INFO("Manager: Max FL rounds (5) reached. Stopping simulation.");
      exportDataFrames();
      Simulator::Stop();
      return;
    }
  }

  // Always schedule next manager check if simulation hasn't stopped
  if (!Simulator::IsFinished()) {
    NS_LOG_INFO("Manager: Scheduling next call in " << managerInterval << " seconds.");
    Simulator::Schedule(Seconds(managerInterval), &manager);
  } else {
    NS_LOG_INFO("Manager: Simulation is finished, not scheduling next call.");
  }
}

void ConfigureDefaults() {
  Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                     TypeIdValue(TcpCubic::GetTypeId()));
  Config::SetDefault("ns3::TcpSocketBase::MinRto",
                     TimeValue(MilliSeconds(200)));
  Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout",
                     TimeValue(Seconds(2)));
  Config::SetDefault("ns3::TcpSocket::SegmentSize",
                     UintegerValue(1448)); // Typical MSS for 1500 MTU
  Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
  uint32_t sndRcvBufSize = 131072 * 10; // 1.3MB
  Config::SetDefault("ns3::TcpSocket::SndBufSize",
                     UintegerValue(sndRcvBufSize));
  Config::SetDefault("ns3::TcpSocket::RcvBufSize",
                     UintegerValue(sndRcvBufSize));
  Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));
  Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(20.0));
  // Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(40.0));
  NS_LOG_INFO("NS-3 default configurations applied.");
}

// Main function
int main(int argc, char *argv[]) {
  // Enable logging for components
  LogComponentEnable("Simulation", LOG_LEVEL_INFO);
  // LogComponentEnable("MyApp", LOG_LEVEL_DEBUG); // Increased logging for MyApp
  LogComponentEnable("Utils", LOG_LEVEL_INFO);
  LogComponentEnable("ClientTypes", LOG_LEVEL_INFO);
  LogComponentEnable("DataFrame", LOG_LEVEL_DEBUG); // DataFrame can be chatty
  LogComponentEnable("Notifications", LOG_LEVEL_INFO); // Keep connection logs visible
  // LogComponentEnable("TcpSocket", LOG_LEVEL_DEBUG); // Increased logging for Sockets
  // LogComponentEnable("TcpSocketBase", LOG_LEVEL_DEBUG); // Even more socket detail
  // LogComponentEnable("Ipv4L3Protocol", LOG_LEVEL_DEBUG); // Logging for network layer issues
  // Uncomment for very detailed debug logs:
  // LogComponentEnable("Simulation", LOG_LEVEL_DEBUG);
  // LogComponentEnable("MyApp", LOG_LEVEL_DEBUG);
  // LogComponentEnable("Utils", LOG_LEVEL_DEBUG);
  // LogComponentEnable("ClientTypes", LOG_LEVEL_DEBUG);
  // LogComponentEnable("DataFrame", LOG_LEVEL_DEBUG);
  // LogComponentEnable("Notifications", LOG_LEVEL_DEBUG);


  // Configure defaults for the simulation
  ConfigureDefaults();
  initializeDataFrames();

  CommandLine cmd;
  cmd.AddValue("algorithm",
               "FL algorithm (ns-3 perspective, less relevant now)", algorithm);
  cmd.Parse(argc, argv);

  // --- Start Python FL API Server ---
  NS_LOG_INFO("Attempting to start Python FL API server...");
  int ret = system("python3 scratch/fl_api.py > fl_api.log 2>&1 &");
  if (ret != 0) {
    NS_LOG_ERROR("ERROR: Failed to start Python FL API server. Exit code: " << ret);
    // return 1; // Can't proceed if API server fails to start
  }
  NS_LOG_INFO("Python FL API server started (hopefully). Waiting for it to "
      "initialize (10s delay)...");
  sleep(10); // Give server time to start up. Robust: poll an endpoint.

  // --- Configure and Initialize Python FL API ---
  NS_LOG_INFO("Configuring Python FL API...");
  json fl_config_payload;
  fl_config_payload["dataset"] = "mnist";
  fl_config_payload["num_clients"] = FL_API_NUM_CLIENTS; // Total UEs in ns-3
  fl_config_payload["clients_per_round"] =
      FL_API_CLIENTS_PER_ROUND; // Max ns-3 can pick
  fl_config_payload["local_epochs"] = 1;
  fl_config_payload["batch_size"] = 32;
  // Add other relevant FL_STATE['config'] parameters from Python API

  int http_code = callPythonApi("/configure", "POST", fl_config_payload.dump()); // <-- Declared http_code here
  if (http_code != 200) {
    NS_LOG_ERROR("ERROR: Failed to configure Python FL API. HTTP Code: " << http_code);
    // return 1;
  } else {
    NS_LOG_INFO("Python FL API configured successfully.");
  }
  sleep(1);

  NS_LOG_INFO("Initializing Python FL API simulation (data loading, initial model)...");
  http_code = callPythonApi("/initialize_simulation", "POST"); // <-- Re-used http_code variable
  if (http_code != 200) {
    NS_LOG_ERROR("ERROR: Failed to initialize Python FL API simulation. HTTP Code: "
        << http_code);
    // return 1;
  } else {
    NS_LOG_INFO("Python FL API simulation initialized successfully.");
    fl_api_initialized = true; // Signal to manager that API is ready
  }
  sleep(1); // Give it a moment

  // --- ns-3 Network Setup (largely unchanged) ---
  Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
  mmwaveHelper->SetEpcHelper(epcHelper);
  mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
  mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
  NS_LOG_INFO("LTE Helper and EPC Helper created and configured.");

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults();

  Ptr<Node> pgw = epcHelper->GetPgwNode();
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  InternetStackHelper internet;
  internet.Install(remoteHostContainer);
  NS_LOG_INFO("PGW and RemoteHost created and InternetStack installed on RemoteHost.");


  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
  p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
  p2ph.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
  remoteHostAddr = internetIpIfaces.GetAddress(1);
  NS_LOG_INFO("Point-to-Point link between PGW and RemoteHost configured. RemoteHost IP: " << remoteHostAddr);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
      ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                             Ipv4Mask("255.0.0.0"), 1);
  NS_LOG_INFO("Static route added on RemoteHost for UE network.");

  enbNodes.Create(numberOfEnbs);
  ueNodes.Create(numberOfUes);
  NS_LOG_INFO("Created " << numberOfEnbs << " eNBs and " << numberOfUes << " UEs.");

MobilityHelper enbmobility;
  // Use RandomRectanglePositionAllocator for random eNB distribution
  Ptr<RandomRectanglePositionAllocator> enbPositionAlloc =
      CreateObject<RandomRectanglePositionAllocator>();
  // Set bounds to the scenario size
  std::string enbBounds = "0|" + std::to_string(scenarioSize) + "|0|" + std::to_string(scenarioSize);
  enbPositionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(scenarioSize) + "]"));
  enbPositionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(scenarioSize) + "]"));
  // Z is 0 by default for 2D allocators, explicitly set if needed, but usually not for typical ground scenarios

  enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  enbmobility.SetPositionAllocator(enbPositionAlloc);
  enbmobility.Install(enbNodes);
  NS_LOG_INFO("eNBs installed with ConstantPositionMobilityModel and random positions within scenario size.");

  // --- UE MOBILITY SETUP (Conditional) ---
  MobilityHelper uemobility;
  if (useStaticClients) {
      NS_LOG_INFO("Installing ConstantPositionMobilityModel for static UEs with random positions.");
      uemobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
      // Use RandomRectanglePositionAllocator for static UE distribution
      Ptr<RandomRectanglePositionAllocator> uePositionAlloc = CreateObject<RandomRectanglePositionAllocator>();
      // Set bounds to the scenario size
      std::string ueBounds = "0|" + std::to_string(scenarioSize) + "|0|" + std::to_string(scenarioSize);
      uePositionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(scenarioSize) + "]"));
      uePositionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(scenarioSize) + "]"));

      uemobility.SetPositionAllocator(uePositionAlloc);
      uemobility.Install(ueNodes);
      NS_LOG_INFO("Static UEs installed with ConstantPositionMobilityModel and random positions within scenario size.");
  } else {
      NS_LOG_INFO("Installing RandomWalk2dMobilityModel for mobile UEs.");
      // Keep Random Walk for UEs, update bounds to use scenarioSize
      std::string walkBounds = "0|" + std::to_string(scenarioSize) + "|0|" + std::to_string(scenarioSize);
      uemobility.SetMobilityModel(
          "ns3::RandomWalk2dMobilityModel", "Mode", StringValue("Time"), "Time",
          StringValue("2s"), "Speed",
          StringValue("ns3::ConstantRandomVariable[Constant=20.0]"), // 20 m/s
          "Bounds", StringValue(walkBounds)); // Bounds based on scenarioSize
      uemobility.Install(ueNodes);
      NS_LOG_INFO("Mobile UEs installed with RandomWalk2dMobilityModel within scenario size bounds.");
  }


  // Install on PGW and RemoteHost too for NetAnim
  enbmobility.Install(pgw); // PGW mobility setup is simpler, can reuse enbmobility
  enbmobility.Install(remoteHost); // RemoteHost is static
  NS_LOG_INFO("Mobility models installed on PGW and RemoteHost for NetAnim.");

  enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
  ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);
  internet.Install(ueNodes);
  epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
  mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs); // Initial attachment
  NS_LOG_INFO("eNB and UE devices installed. UE IP addresses assigned. UEs attached to closest eNB.");

  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<Node> ueNode = ueNodes.Get(i);
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(),
                                     1);
  }
  NS_LOG_INFO("Static routes set for UEs.");

  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();
    // Connect to the RSRP/SINR trace source using the correctly-signatured
    // callback
    uePhy->TraceConnectWithoutContext(
        "ReportCurrentCellRsrpSinr",
        MakeCallback<void, uint16_t, uint16_t, double, double, uint8_t>(
            &ReportUeSinrRsrp));
  }
  NS_LOG_INFO("RSRP/SINR trace sources connected for UEs.");

  Ptr<FlowMonitor> monitor = flowmon.InstallAll();
  NS_LOG_INFO("FlowMonitor installed.");

  // --- Schedule ns-3 simulation events ---
  Simulator::Schedule(Seconds(2.0),
                      &manager); // Start manager after a brief delay
  Simulator::Schedule(Seconds(1.0), &networkInfo,
                      monitor); // Start network info collection
  NS_LOG_INFO("Manager and networkInfo functions scheduled.");

  AnimationInterface anim("fl_api_mmwave_animation.xml");
  // anim.SetMobilityPollInterval(Seconds(1)); // Optional: control netanim
  // update rate
  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    anim.UpdateNodeDescription(ueNodes.Get(i), "UE");
    anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0);
  }
  for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
    anim.UpdateNodeDescription(enbNodes.Get(i), "ENB");
    anim.UpdateNodeColor(enbNodes.Get(i), 0, 255, 0);
  }
  anim.UpdateNodeDescription(remoteHost, "RH_FL_Server");
  anim.UpdateNodeColor(remoteHost, 0, 0, 255);
  anim.UpdateNodeDescription(pgw, "PGW");
  anim.UpdateNodeColor(pgw, 0, 0, 255);
  NS_LOG_INFO("NetAnim configuration complete.");

  // Connection notification callbacks (can be useful for debugging)
  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedEnb));
  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedUe));
  NS_LOG_INFO("LTE ConnectionEstablished trace sources connected.");

  Simulator::Stop(Seconds(simStopTime));
  NS_LOG_INFO("Starting ns-3 Simulation. Simulation will stop at " << simStopTime << "s.");
  Simulator::Run();
  NS_LOG_INFO("ns-3 Simulation Finished.");

  // --- Clean up ---
  // Terminate Python API server (optional, could be done manually or via kill
  // command) Find PID of "python3 scratch/sim/fl_api.py" and kill it.
  system("pkill -f 'python3 scratch/sim/fl_api.py'"); // Might be too
  // aggressive if other python3 scripts are running
  NS_LOG_INFO("Remember to manually stop the Python FL API server if it's still "
      "running (e.g., using 'pkill -f fl_api.py').");

  exportDataFrames(); // Final export
  Simulator::Destroy();
  NS_LOG_INFO("Simulator Destroyed.");
  return 0;
}