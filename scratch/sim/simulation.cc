// Filename: scratch/sim/simulation.cc
// Macro for logging
#define LOG(x) std::cout << x << std::endl

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

// Using declarations for convenience
using namespace ns3;
using json = nlohmann::json;

// Define the simulation logging component
NS_LOG_COMPONENT_DEFINE("Simulation");

// Global constants
static constexpr double simStopTime = 1200.0;
static constexpr int numberOfUes = 10; // Reduced for faster testing
static constexpr int numberOfEnbs = 2; // Reduced for faster testing
static constexpr int numberOfParticipatingClients =
    5; // Max clients per round for FL
static constexpr int scenarioSize = 1000;
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
Time timeout = Seconds(120); // ns-3 round timeout for model transfers
static double constexpr managerInterval =
    0.1; // ns-3 manager check interval, make it larger e.g. 1.0

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

  LOG("callPythonApi: PREPARING to execute CURL for endpoint "
      << endpoint << " at " << Simulator::Now().GetSeconds()
      << "s. Command: " << command.str());

  std::fflush(stdout); // Force flush output before potentially blocking popen

  char buffer[128];
  std::string result_str = "";
  FILE *pipe = popen(command.str().c_str(), "r");
  if (!pipe) {
    LOG("ERROR: popen() FAILED for command: "
        << command.str() << " at " << Simulator::Now().GetSeconds() << "s.");
    std::fflush(stdout);
    return -1;
  }
  LOG("callPythonApi: popen successful, READING from pipe for "
      << endpoint << " at " << Simulator::Now().GetSeconds() << "s.");
  std::fflush(stdout);
  try {
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      result_str += buffer;
    }
  }

  catch (const std::exception &e) {
    LOG("ERROR: Exception while reading from pipe for "
        << endpoint << ": " << e.what() << " at "
        << Simulator::Now().GetSeconds() << "s.");
    std::fflush(stdout);
    pclose(pipe);
    return -2;
  } catch (...) {
    LOG("ERROR: Unknown exception while reading from pipe for "
        << endpoint << " at " << Simulator::Now().GetSeconds() << "s.");
    std::fflush(stdout);
    pclose(pipe);
    return -3;
  }

  LOG("callPythonApi: FINISHED reading from pipe for "
      << endpoint << ". Raw result_str: '" << result_str << "'"
      << " at " << Simulator::Now().GetSeconds() << "s.");
  std::fflush(stdout);

  int status = pclose(pipe);
  LOG("callPythonApi: pclose status for "
      << endpoint << ": " << status << " at " << Simulator::Now().GetSeconds()
      << "s.");
  std::fflush(stdout);

  //   catch (...) {
  //     pclose(pipe);
  //     LOG("ERROR: Exception while reading from pipe");
  //     return -1;
  //   }
  //   int status = pclose(pipe);
  if (status == -1) {
    LOG("ERROR: pclose failed or command not found. Status: " << status);
    return -1;
  } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
    LOG("ERROR: Curl command exited with status "
        << WEXITSTATUS(status) << ". HTTP code: " << result_str);
    // result_str might contain the http code even if curl itself had an error
    // code (e.g. connection refused)
  }

  try {
    return std::stoi(result_str);
  } catch (const std::invalid_argument &ia) {
    LOG("ERROR: Invalid argument for stoi: '" << result_str << "'");
    return -1; // Or some other error code
  } catch (const std::out_of_range &oor) {
    LOG("ERROR: Out of range for stoi: '" << result_str << "'");
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
  std::vector<std::string> throughput_columns = {"time", "tx_throughput",
                                                 "rx_throughput"};

  for (const auto &column : accuracy_columns)
    accuracy_df.addColumn(column);
  for (const auto &column : participation_columns)
    participation_df.addColumn(column);
  for (const auto &column : throughput_columns)
    throughput_df.addColumn(column);
}

std::pair<double, double> getRsrpSinr(uint32_t nodeIdx) {
  Ptr<NetDevice> ueDevice = ueDevs.Get(nodeIdx);
  if (!ueDevice)
    return {0.0, 0.0};
  auto lteUeNetDevice = ueDevice->GetObject<LteUeNetDevice>();
  if (!lteUeNetDevice)
    return {0.0, 0.0};
  auto rrc = lteUeNetDevice->GetRrc();
  // if (!rrc || !rrc->IsConnected()) return {0.0, 0.0}; // Check if RRC is
  // valid and connected

  if (!rrc || (rrc->GetState() != LteUeRrc::CONNECTED_NORMALLY &&
               rrc->GetState() != LteUeRrc::CONNECTED_HANDOVER)) {
    // Alternative check: if (!rrc || rrc->GetRnti() ==
    // LteUeRrc::UNINITIALIZED_RNTI)
    return {0.0, 0.0}; // Not connected or RRC not available
  }
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
  return {rsrp, sinr};
}

// Fills clientsInfoGlobal with current data for ALL UEs
void updateAllClientsGlobalInfo() {
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
  }
}

// Selects clients based on ns-3 criteria for the current round
// Populates `selectedClientsForCurrentRound`
void selectNs3ManagedClients(int n_to_select) {
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

  for (int i = 0; i < n_to_select && (long unsigned int)i < candidates.size();
       ++i) {
    ClientModels selected_client = candidates[i];
    selected_client.selected =
        true; // Mark as selected for ns-3 comms simulation
    selectedClientsForCurrentRound.push_back(selected_client);
  }
  LOG("ns-3 selected " << selectedClientsForCurrentRound.size()
                       << " clients for FL round " << roundNumber);
}

// This function is now responsible for:
// 1. Telling Python API which clients ns-3 has selected.
// 2. Python API runs the full FL round (training + aggregation).
// 3. Logging results from Python API.
// 4. Preparing for ns-3 to simulate model uploads from these selected clients.
bool triggerAndProcessFLRoundInApi() {
  LOG("=================== Triggering FL Round "
      << roundNumber << " in Python API at " << Simulator::Now().GetSeconds()
      << "s ===================");
  std::fflush(stdout); // Ensure log is flushed

  if (selectedClientsForCurrentRound.empty() &&
      FL_API_CLIENTS_PER_ROUND >
          0) // Check if ns-3 wants to select but couldn't
  {
    LOG("No clients were selected by ns-3 for this round. Skipping API call "
        "for /run_round.");
    std::fflush(stdout);
    // If no clients are selected by ns-3, we might still want to "complete" the
    // API part of the round or decide that the API round shouldn't run. For
    // now, let's assume if ns-3 selects 0, the API round for training is
    // skipped for these 0. The Python API itself, if called with an empty
    // client_indices, will sample its own. This behavior needs to be aligned.
    // Let's assume for now we only proceed if ns-3 selected clients.
    if (FL_API_CLIENTS_PER_ROUND > 0)
      return false; // No clients to send to API for training
  }

  // TEST CALL FOR ROUND 2
  if (roundNumber == 2) {
    LOG("Attempting a TEST CURL to /ping before FL API round 2 /run_round call "
        "at "
        << Simulator::Now().GetSeconds() << "s");
    std::fflush(stdout);
    int test_http_code = callPythonApi("/ping", "GET", "", "ping_response.txt");
    LOG("TEST /ping call HTTP code: " << test_http_code << " at "
                                      << Simulator::Now().GetSeconds() << "s");
    std::fflush(stdout);
    if (test_http_code != 200) {
      LOG("ERROR: Test /ping call FAILED. Aborting before /run_round for round "
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
  LOG("Calling /run_round for round "
      << roundNumber << " with payload: " << client_indices_payload.dump());
  std::fflush(stdout);
  int http_code = callPythonApi("/run_round", "POST",
                                client_indices_payload.dump(), response_file);

  if (http_code == 200) {
    LOG("Python API /run_round call successful for round " << roundNumber);
    std::ifstream ifs(response_file);
    if (ifs.is_open()) {
      json response_json;
      try {
        ifs >> response_json;
        LOG("Python API Response: " << response_json.dump(2));
        // Log metrics to accuracy_df
        accuracy_df.addRow(
            {Simulator::Now().GetSeconds(), roundNumber,
             response_json.value("global_test_accuracy", 0.0),
             response_json.value("global_test_loss", 0.0),
             response_json.value("avg_client_accuracy", 0.0),
             response_json.value("avg_client_loss", 0.0),
             response_json.value("round_duration_seconds", 0.0)});
      } catch (json::parse_error &e) {
        LOG("ERROR: Failed to parse Python API response JSON: " << e.what());
        return false;
      }
    } else {
      LOG("ERROR: Could not open response file: " << response_file);
      return false;
    }
    return true;
  } else {
    LOG("ERROR: Python API /run_round call failed. HTTP Code: " << http_code);
    std::ifstream ifs(response_file);
    if (ifs.is_open()) {
      json error_json;
      try {
        ifs >> error_json;
        LOG("Python API Error Response: " << error_json.dump(2));
      } catch (json::parse_error &e) {
        LOG("ERROR: Failed to parse Python API error JSON: " << e.what());
      }
    }
    return false;
  }
}

void sendModelsToServer() { // Uses selectedClientsForCurrentRound
  LOG("ns-3: Simulating model uploads for selected clients.");
  for (const auto &client_model_info : selectedClientsForCurrentRound) {
    // .selected flag is already true from selectNs3ManagedClients
    // nodeModelSize and nodeTrainingTime are from clientsInfoGlobal (default
    // values)
    LOG("Client " << client_model_info.node->GetId()
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
  return Simulator::Now() - roundStartTimeNs3Comms > timeout;
}

void logRoundTimeout() {
  LOG("ns-3 Comms Round timed out. Successful transfers: "
      << endOfStreamTimes.size() << "/"
      << selectedClientsForCurrentRound.size());
}

void addParticipationToDataFrame() {
  participation_df.addRow({Simulator::Now().GetSeconds(), roundNumber,
                           (uint32_t)selectedClientsForCurrentRound.size(),
                           (uint32_t)endOfStreamTimes.size()});
}

// This finalize is for the ns-3 communication part of the round
void finalizeNs3CommsPhase() {
  LOG("ns-3 Comms phase for round "
      << roundNumber << " finished at " << Simulator::Now().GetSeconds()
      << ". Successful transfers: " << endOfStreamTimes.size() << "/"
      << selectedClientsForCurrentRound.size());
  // The actual FL aggregation was already done by Python API's /run_round.
  // No separate aggregation() call needed here from ns-3 unless it's for a
  // different purpose.

  addParticipationToDataFrame();
  roundCleanup(); // Clears ns-3 apps and endOfStreamTimes
}

void startNewFLRound(
    Time &roundStartTimeNs3CommsParam) // Parameter renamed to avoid conflict
                                       // with static
{
  roundNumber++;
  LOG("StartNewFLRound: Beginning for FL Round "
      << roundNumber << " at " << Simulator::Now().GetSeconds() << "s.");

  selectNs3ManagedClients(FL_API_CLIENTS_PER_ROUND);
  LOG("StartNewFLRound: ns-3 selected " << selectedClientsForCurrentRound.size()
                                        << " clients.");

  if (selectedClientsForCurrentRound.empty() && FL_API_CLIENTS_PER_ROUND > 0) {
    LOG("StartNewFLRound: No clients were selected by ns-3 (e.g., due to SINR "
        "or other criteria, or no eligible UEs). Skipping API call and ns-3 "
        "comms for this round.");
    roundFinished = true; // Mark as finished to allow manager to proceed
    // No need to increment roundNumber here again, manager will loop.
    return;
  }
  // If FL_API_CLIENTS_PER_ROUND is 0, selectedClientsForCurrentRound will be
  // empty. The Python API /run_round expects client_indices. If empty, it
  // should handle it or ns-3 should not call. Current Python API /run_round
  // samples clients if client_indices is not provided or empty. For clarity,
  // let's ensure ns-3 always provides the list of clients it selected, even if
  // empty.

  LOG("StartNewFLRound: Triggering FL round in Python API for round "
      << roundNumber);
  bool api_success = triggerAndProcessFLRoundInApi(); // This calls Python API

  if (api_success) {
    LOG("StartNewFLRound: Python API call successful for round "
        << roundNumber);
    if (!selectedClientsForCurrentRound.empty()) {
      LOG("StartNewFLRound: Scheduling ns-3 model uploads for "
          << selectedClientsForCurrentRound.size() << " clients.");
      sendModelsToServer(); // Schedules ns-3 MyApp instances
      roundStartTimeNs3CommsParam =
          Simulator::Now();  // Mark start of ns-3 communication phase
      roundFinished = false; // ns-3 communication phase now active
      LOG("StartNewFLRound: ns-3 comms phase started for round "
          << roundNumber << ". roundFinished set to false.");
    } else {
      LOG("StartNewFLRound: Python API call successful, but no clients "
          "selected by ns-3 for simulated upload. Marking round as "
          "(comms-wise) finished.");
      roundFinished = true; // No ns-3 comms to simulate
    }
  } else {
    LOG("StartNewFLRound: Python API call FAILED for round "
        << roundNumber << ". Skipping ns-3 comms phase.");
    roundFinished = true; // Mark as finished to allow manager to try next FL
                          // round attempt (or stop if max rounds)
  }
}

void exportDataFrames() {
  accuracy_df.toCsv("accuracy_fl_api.csv");
  participation_df.toCsv("clientParticipation_fl_api.csv");
  throughput_df.toCsv("throughput_fl_api.csv");
}

void manager() {
  static Time roundStartTimeNs3Comms =
      Simulator::Now(); // Initialize to current time at first call
  LOG("Manager called at " << Simulator::Now().GetSeconds()
                           << "s. RoundNumber: " << roundNumber
                           << ", roundFinished (ns-3 comms): "
                           << roundFinished);

  if (!fl_api_initialized) {
    LOG("FL API not yet initialized by main. Manager waiting.");
    Simulator::Schedule(Seconds(5.0), &manager);
    return;
  }

  if (!roundFinished) {
    LOG("Manager: ns-3 communication phase for round " << roundNumber
                                                       << " is ongoing.");
    if (isRoundTimedOut(roundStartTimeNs3Comms)) {
      LOG("Manager: Round " << roundNumber << " ns-3 comms timed out.");
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
          LOG("Manager: All selected clients ("
              << endOfStreamTimes.size() << "/"
              << selectedClientsForCurrentRound.size()
              << ") completed ns-3 transmissions for round " << roundNumber);
        } else if (selectedClientsForCurrentRound.empty()) {
          LOG("Manager: No clients were selected for ns-3 comms in round "
              << roundNumber << ", considering comms phase complete.");
        }
        roundFinished = true;
      } else {
        LOG("Manager: Waiting for "
            << selectedClientsForCurrentRound.size() - endOfStreamTimes.size()
            << " more clients to finish ns-3 comms for round " << roundNumber);
      }
    }

    if (roundFinished) {
      LOG("Manager: Finalizing ns-3 comms phase for round " << roundNumber);
      finalizeNs3CommsPhase(); // Logs, adds to participation_df, cleans up apps
    }
  }

  if (roundFinished) {
    LOG("Manager: ns-3 communication phase for round "
        << roundNumber << " is finished or was skipped.");
    if (roundNumber < 5) // Limit total FL rounds for testing
    {
      LOG("Manager: Attempting to start new FL round (will be round "
          << roundNumber + 1 << ")");
      startNewFLRound(roundStartTimeNs3Comms); // This will attempt to set
                                               // roundFinished=false
    } else {
      LOG("Manager: Max FL rounds (5) reached. Stopping simulation.");
      exportDataFrames();
      Simulator::Stop();
      return;
    }
  }

  // Always schedule next manager check if simulation hasn't stopped
  if (!Simulator::IsFinished()) {
    LOG("Manager: Scheduling next call.");
    Simulator::Schedule(Seconds(1.0), &manager);
  } else {
    LOG("Manager: Simulation is finished, not scheduling next call.");
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
}

// Main function
int main(int argc, char *argv[]) {
  // Configure defaults for the simulation
  ConfigureDefaults();
  initializeDataFrames();

  CommandLine cmd;
  cmd.AddValue("algorithm",
               "FL algorithm (ns-3 perspective, less relevant now)", algorithm);
  cmd.Parse(argc, argv);

  // --- Start Python FL API Server ---
  LOG("Attempting to start Python FL API server...");
  // Fork and exec to run python script in background might be cleaner for
  // production but system() with & is simpler for this context. Make sure
  // fl_api.py is executable or use "python3 scratch/sim/fl_api.py &"
  int ret = system("python3 scratch/fl_api.py > fl_api.log 2>&1 &");
  if (ret != 0) {
    LOG("ERROR: Failed to start Python FL API server. Exit code: " << ret);
    // return 1; // Can't proceed if API server fails to start
  }
  LOG("Python FL API server started (hopefully). Waiting for it to "
      "initialize...");
  sleep(10); // Give server time to start up. Robust: poll an endpoint.

  // --- Configure and Initialize Python FL API ---
  LOG("Configuring Python FL API...");
  json fl_config_payload;
  fl_config_payload["dataset"] = "mnist";
  fl_config_payload["num_clients"] = FL_API_NUM_CLIENTS; // Total UEs in ns-3
  fl_config_payload["clients_per_round"] =
      FL_API_CLIENTS_PER_ROUND; // Max ns-3 can pick
  fl_config_payload["local_epochs"] = 1;
  fl_config_payload["batch_size"] = 32;
  // Add other relevant FL_STATE['config'] parameters from Python API

  int http_code = callPythonApi("/configure", "POST", fl_config_payload.dump());
  if (http_code != 200) {
    LOG("ERROR: Failed to configure Python FL API. HTTP Code: " << http_code);
    // return 1;
  } else {
    LOG("Python FL API configured successfully.");
  }
  sleep(1);

  LOG("Initializing Python FL API simulation (data loading, initial model)...");
  http_code = callPythonApi("/initialize_simulation", "POST");
  if (http_code != 200) {
    LOG("ERROR: Failed to initialize Python FL API simulation. HTTP Code: "
        << http_code);
    // return 1;
  } else {
    LOG("Python FL API simulation initialized successfully.");
    fl_api_initialized = true; // Signal to manager that API is ready
  }
  sleep(1); // Give it a moment

  // --- ns-3 Network Setup (largely unchanged) ---
  Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
  mmwaveHelper->SetEpcHelper(epcHelper);
  mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
  mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults();

  Ptr<Node> pgw = epcHelper->GetPgwNode();
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  InternetStackHelper internet;
  internet.Install(remoteHostContainer);

  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
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
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                             Ipv4Mask("255.0.0.0"), 1);

  enbNodes.Create(numberOfEnbs);
  ueNodes.Create(numberOfUes);

  MobilityHelper enbmobility, uemobility;
  Ptr<ListPositionAllocator> enbPositionAlloc =
      CreateObject<ListPositionAllocator>();
  for (int i = 0; i < numberOfEnbs; ++i) {
    enbPositionAlloc->Add(Vector(i * 200, 0, 0)); // Spread eNBs
  }
  enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  enbmobility.SetPositionAllocator(enbPositionAlloc);
  enbmobility.Install(enbNodes);

  // Random walk for UEs
  uemobility.SetMobilityModel(
      "ns3::RandomWalk2dMobilityModel", "Mode", StringValue("Time"), "Time",
      StringValue("2s"), "Speed",
      StringValue("ns3::ConstantRandomVariable[Constant=20.0]"), // 20 m/s
      "Bounds", StringValue("0|1000|0|1000")); // 1km x 1km area
  uemobility.Install(ueNodes);

  // Install on PGW and RemoteHost too for NetAnim
  enbmobility.Install(pgw);
  enbmobility.Install(remoteHost);

  enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
  ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);
  internet.Install(ueNodes);
  epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
  mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs); // Initial attachment

  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<Node> ueNode = ueNodes.Get(i);
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(),
                                     1);
  }

  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();
    // Connect to the RSRP/SINR trace source using the correctly-signatured
    // callback
    uePhy->TraceConnectWithoutContext(
        "ReportCurrentCellRsrpSinr",
        MakeCallback<void, uint16_t, uint16_t, double, double, uint8_t>(
            &ReportUeSinrRsrp));

    // COMMENT OUT or DELETE the following problematic line:
    // uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr",
    //                               MakeCallback(&ReportUePhyMetricsFromTrace));
  }
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();

  // --- Schedule ns-3 simulation events ---
  Simulator::Schedule(Seconds(2.0),
                      &manager); // Start manager after a brief delay
  Simulator::Schedule(Seconds(1.0), &networkInfo,
                      monitor); // Start network info collection

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

  // Connection notification callbacks (can be useful for debugging)
  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedEnb));
  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedUe));

  Simulator::Stop(Seconds(simStopTime));
  LOG("Starting ns-3 Simulation.");
  Simulator::Run();
  LOG("ns-3 Simulation Finished.");

  // --- Clean up ---
  // Terminate Python API server (optional, could be done manually or via kill
  // command) Find PID of "python3 scratch/sim/fl_api.py" and kill it.
  // system("pkill -f 'python3 scratch/sim/fl_api.py'"); // Might be too
  // aggressive if other python3 scripts are running
  LOG("Remember to manually stop the Python FL API server if it's still "
      "running.");

  exportDataFrames(); // Final export
  Simulator::Destroy();
  return 0;
}