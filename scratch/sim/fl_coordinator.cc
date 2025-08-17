#include "fl_coordinator.h"
#include "metrics_collector.h"
#include "network_utils.h"
#include "network_setup.h"
#include "ns3/log.h"
#include "json.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <unistd.h>

NS_LOG_COMPONENT_DEFINE("FLCoordinator");

using json = nlohmann::json;

std::string FL_API_BASE_URL = "";
const int FL_API_NUM_CLIENTS = 10;
const int FL_API_CLIENTS_PER_ROUND = 5;
Time timeout = Seconds(50);
static double constexpr managerInterval = 1.0;
static bool roundFinished = true;
int roundNumber = 0;
static bool fl_api_initialized = false;

std::vector<NodesIps> nodesIPs;
std::vector<ClientModels> clientsInfoGlobal;
std::vector<ClientModels> selectedClientsForCurrentRound;

std::string FLCoordinator::getEnvVar(const std::string &key, const std::string &default_val) {
    const char* val = std::getenv(key.c_str());
    if (val == nullptr) {
        NS_LOG_INFO("Environment variable '" << key << "' not found. Using default value: '" << default_val << "'.");
        return default_val;
    }
    NS_LOG_INFO("Found environment variable '" << key << "'. Value: '" << val << "'.");
    return std::string(val);
}

std::string FLCoordinator::exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    NS_LOG_DEBUG("exec: Running command: " << cmd);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
         NS_LOG_ERROR("exec: popen() failed: " << strerror(errno));
         throw std::runtime_error("popen() failed!");
    }
    try {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
    } catch (...) {
        pclose(pipe);
        NS_LOG_ERROR("exec: Exception occurred while reading from pipe.");
        throw;
    }
    int exit_code = pclose(pipe);
     if (exit_code == -1) {
        NS_LOG_ERROR("exec: pclose() failed: " << strerror(errno));
    } else if (WIFEXITED(exit_code) && WEXITSTATUS(exit_code) != 0) {
        NS_LOG_ERROR("exec: Command exited with non-zero status " << WEXITSTATUS(exit_code) << " for command: " << cmd);
    } else if (WIFSIGNALED(exit_code)) {
         NS_LOG_ERROR("exec: Command killed by signal " << WTERMSIG(exit_code) << " for command: " << cmd);
    }
    NS_LOG_DEBUG("exec: Command finished. Output: " << result);
    return result;
}

int FLCoordinator::callPythonApi(const std::string &endpoint,
                  const std::string &method,
                  const std::string &data,
                  const std::string &output_file) {
  std::stringstream command;
  command << "curl --max-time 20 --connect-timeout 5 -s -o ";
  if (output_file.empty()) {
    command << "/dev/null";
  } else {
    command << output_file;
  }
  command << " -w \"%{http_code}\"";

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
    NS_LOG_DEBUG("callPythonApi: Payload: "
                 << data.substr(0, 200) << (data.length() > 200 ? "..." : ""));
  }

  std::fflush(stdout);

  char buffer[128];
  std::string result_str = "";
  FILE *pipe = popen(command.str().c_str(), "r");
  if (!pipe) {
    NS_LOG_ERROR("callPythonApi: popen() FAILED for command: "
                 << command.str() << " at " << Simulator::Now().GetSeconds()
                 << "s. Error: " << strerror(errno));
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
                 << endpoint << " at " << Simulator::Now().GetSeconds()
                 << "s.");
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
              << endpoint << ": " << status << " at "
              << Simulator::Now().GetSeconds()
              << "s. Curl result: '" << result_str << "'");
  std::fflush(stdout);

  if (status == -1) {
    NS_LOG_ERROR("callPythonApi: pclose failed or command not found. Status: "
                 << status << ". Error: " << strerror(errno));
    return -1;
  } else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
    NS_LOG_ERROR("callPythonApi: Curl command exited with status "
                 << WEXITSTATUS(status) << ". HTTP code: '" << result_str << "'");
  }

  try {
    return std::stoi(result_str);
  } catch (const std::invalid_argument &ia) {
    NS_LOG_ERROR("callPythonApi: Invalid argument for stoi: '" << result_str
                                                               << "'");
    return -1;
  } catch (const std::out_of_range &oor) {
    NS_LOG_ERROR("callPythonApi: Out of range for stoi: '" << result_str
                                                           << "'");
    return -1;
  }
}

bool FLCoordinator::initializeFlApi() {
    FL_API_BASE_URL = getEnvVar("FL_API_URL", "http://127.0.0.1:5005");
    
    NS_LOG_INFO("Configuring Python FL API...");
    int api_port = 5005;
    json fl_config_payload;
    fl_config_payload["dataset"] = "mnist";
    fl_config_payload["num_clients"] = FL_API_NUM_CLIENTS;
    fl_config_payload["clients_per_round"] = FL_API_CLIENTS_PER_ROUND;
    fl_config_payload["local_epochs"] = 1;
    fl_config_payload["batch_size"] = 32;
    fl_config_payload["port"] = api_port;

    int http_code = callPythonApi("/configure", "POST", fl_config_payload.dump());
    if (http_code != 200) {
        NS_LOG_ERROR("ERROR: Failed to configure Python FL API. HTTP Code: " << http_code);
        return false;
    } else {
        NS_LOG_INFO("Python FL API configured successfully.");
    }
    sleep(1);

    NS_LOG_INFO("Initializing Python FL API simulation (data loading, initial model)...");
    http_code = callPythonApi("/initialize_simulation", "POST");
    if (http_code != 200) {
        NS_LOG_ERROR("ERROR: Failed to initialize Python FL API simulation. HTTP Code: " << http_code);
        return false;
    } else {
        NS_LOG_INFO("Python FL API simulation initialized successfully.");
        fl_api_initialized = true;
    }
    sleep(1);
    return true;
}

void FLCoordinator::selectNs3ManagedClients(int n_to_select) {
  NS_LOG_INFO("Selecting " << n_to_select << " clients for FL round "
                           << roundNumber << " based on ns-3 criteria.");
  selectedClientsForCurrentRound.clear();
  MetricsCollector::updateAllClientsGlobalInfo();

  std::vector<ClientModels> candidates = clientsInfoGlobal;
  std::sort(candidates.begin(), candidates.end(),
            [](const ClientModels &a, const ClientModels &b) {
              return a.sinr > b.sinr;
            });

  int actual_selected_count = 0;
  for (int i = 0; i < n_to_select && (long unsigned int)i < candidates.size(); ++i) {
    if (candidates[i].sinr > 0.001 || candidates[i].rsrp < 0.0) {
      ClientModels selected_client = candidates[i];
      selected_client.selected = true;
      selectedClientsForCurrentRound.push_back(selected_client);
      actual_selected_count++;
      NS_LOG_DEBUG("  Selected client " << selected_client.node->GetId()
                                        << " (SINR: " << selected_client.sinr
                                        << " dB, RSRP: " << selected_client.rsrp
                                        << " dBm)");
    } else {
      NS_LOG_DEBUG("  Skipping client "
                   << candidates[i].node->GetId() << " due to low SINR ("
                   << candidates[i].sinr << " dB) or RSRP ("
                   << candidates[i].rsrp << " dBm).");
    }
  }
  NS_LOG_INFO("ns-3 selected " << actual_selected_count << " clients (out of "
                               << n_to_select << " requested) for FL round "
                               << roundNumber);
  if (actual_selected_count == 0 && n_to_select > 0) {
    NS_LOG_WARN("No eligible clients were selected by ns-3 for this round, "
                "possibly due to poor network conditions for all UEs.");
  }
}

bool FLCoordinator::triggerAndProcessFLRoundInApi() {
  NS_LOG_INFO("=================== Triggering FL Round "
              << roundNumber << " in Python API at "
              << Simulator::Now().GetSeconds() << "s ===================");
  std::fflush(stdout);

  if (selectedClientsForCurrentRound.empty() && FL_API_CLIENTS_PER_ROUND > 0) {
    NS_LOG_INFO("No clients were selected by ns-3 for this round. Skipping API call for /run_round.");
    std::fflush(stdout);
    if (FL_API_CLIENTS_PER_ROUND > 0)
      return false;
  }

  json client_indices_payload;
  std::vector<int> indices_list;
  for (const auto &client : selectedClientsForCurrentRound) {
    indices_list.push_back(static_cast<int>(client.node->GetId()));
  }
  client_indices_payload["client_indices"] = indices_list;

  std::string response_file = "fl_round_response.json";
  NS_LOG_INFO("Calling /run_round for round "
              << roundNumber << " with payload (first 100 chars): "
              << client_indices_payload.dump().substr(0, 100) << "...");
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
        NS_LOG_INFO("Python API Response (first 200 chars): "
                    << response_json.dump(2).substr(0, 200) << "...");
        
        accuracy_df.addRow(
            {Simulator::Now().GetSeconds(), roundNumber,
             response_json.value("global_test_accuracy", 0.0),
             response_json.value("global_test_loss", 0.0),
             response_json.value("avg_client_accuracy", 0.0),
             response_json.value("avg_client_loss", 0.0),
             response_json.value("round_duration_seconds", 0.0)});
        NS_LOG_INFO("Accuracy data added to DataFrame for round " << roundNumber);

        auto client_perf_details = response_json.value("simulated_client_performance", json::object());
        NS_LOG_INFO("Updating selected client info with simulated values from API response ("
                    << client_perf_details.size() << " clients)...");
        for (auto const &[client_id_str, perf_data] : client_perf_details.items()) {
          try {
            int client_id = std::stoi(client_id_str);
            for (auto &client_model_info : selectedClientsForCurrentRound) {
              if (client_model_info.node->GetId() == (uint32_t)client_id) {
                client_model_info.nodeTrainingTime =
                    perf_data.value("simulated_training_time_ms",
                                    client_model_info.nodeTrainingTime);
                client_model_info.nodeModelSize =
                    perf_data.value("simulated_model_size_bytes",
                                    client_model_info.nodeModelSize);
                NS_LOG_DEBUG("  Updated client "
                             << client_id << ": training_time="
                             << client_model_info.nodeTrainingTime
                             << "ms, model_size="
                             << client_model_info.nodeModelSize << " bytes.");
                break;
              }
            }
          } catch (const std::invalid_argument &ia) {
            NS_LOG_ERROR("  Failed to parse client ID string: " << client_id_str);
          }
        }

      } catch (json::parse_error &e) {
        NS_LOG_ERROR("ERROR: Failed to parse Python API response JSON from '"
                     << response_file << "': " << e.what());
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
        NS_LOG_ERROR("ERROR: Failed to parse Python API error JSON from '"
                     << response_file << "': " << e.what());
      }
    }
    return false;
  }
}

void FLCoordinator::sendModelsToServer() {
  NS_LOG_INFO("ns-3: Simulating model uploads for "
              << selectedClientsForCurrentRound.size() << " selected clients.");
  if (selectedClientsForCurrentRound.empty()) {
    NS_LOG_INFO("  No clients selected for model upload in ns-3 this round. Skipping send.");
    return;
  }

  for (const auto &client_model_info : selectedClientsForCurrentRound) {
    NS_LOG_INFO("  Client " << client_model_info.node->GetId()
                            << " scheduling ns-3 send model of size "
                            << client_model_info.nodeModelSize << " bytes "
                            << "after " << client_model_info.nodeTrainingTime
                            << "ms pseudo-training time.");

    Simulator::Schedule(MilliSeconds(client_model_info.nodeTrainingTime),
                        &NetworkUtils::sendStream, client_model_info.node,
                        remoteHostContainer.Get(0),
                        client_model_info.nodeModelSize);
  }
}

bool FLCoordinator::isRoundTimedOut(Time roundStartTimeNs3Comms) {
  bool timedOut = Simulator::Now() - roundStartTimeNs3Comms > timeout;
  if (timedOut) {
    NS_LOG_WARN("isRoundTimedOut: ns-3 Comms phase for round "
                << roundNumber << " has timed out at "
                << Simulator::Now().GetSeconds() << "s.");
  } else {
    NS_LOG_DEBUG("isRoundTimedOut: ns-3 Comms phase for round "
                 << roundNumber << " is not yet timed out. Current duration: "
                 << (Simulator::Now() - roundStartTimeNs3Comms).GetSeconds()
                 << "s.");
  }
  return timedOut;
}

void FLCoordinator::logRoundTimeout() {
  NS_LOG_WARN("ns-3 Comms Round timed out for round "
              << roundNumber
              << ". Successful transfers: " << endOfStreamTimes.size() << "/"
              << selectedClientsForCurrentRound.size() << " clients.");
}

void FLCoordinator::addParticipationToDataFrame() {
  NS_LOG_INFO("Adding participation data to DataFrame for round " << roundNumber << ".");
  participation_df.addRow({Simulator::Now().GetSeconds(), roundNumber,
                           (uint32_t)selectedClientsForCurrentRound.size(),
                           (uint32_t)endOfStreamTimes.size()});
  NS_LOG_INFO("  Recorded selected: " << selectedClientsForCurrentRound.size()
                                      << ", completed ns-3 comms: "
                                      << endOfStreamTimes.size());
}

void FLCoordinator::finalizeNs3CommsPhase() {
  NS_LOG_INFO("ns-3 Comms phase for round "
              << roundNumber << " finished at " << Simulator::Now().GetSeconds()
              << ". Successful transfers: " << endOfStreamTimes.size() << "/"
              << selectedClientsForCurrentRound.size());

  addParticipationToDataFrame();
  NetworkUtils::roundCleanup();
  NS_LOG_INFO("ns-3 Comms phase cleanup complete for round " << roundNumber << ".");
}

void FLCoordinator::startNewFLRound(Time &roundStartTimeNs3CommsParam) {
  roundNumber++;
  NS_LOG_INFO("StartNewFLRound: Beginning for FL Round "
              << roundNumber << " at " << Simulator::Now().GetSeconds()
              << "s.");

  selectNs3ManagedClients(FL_API_CLIENTS_PER_ROUND);
  NS_LOG_INFO("StartNewFLRound: ns-3 selected "
              << selectedClientsForCurrentRound.size()
              << " clients for this round.");

  if (selectedClientsForCurrentRound.empty() && FL_API_CLIENTS_PER_ROUND > 0) {
    NS_LOG_INFO("StartNewFLRound: No clients were selected by ns-3. Skipping API call and ns-3 comms for this round.");
    roundFinished = true;
    return;
  }

  NS_LOG_INFO("StartNewFLRound: Triggering FL round in Python API for round " << roundNumber);
  bool api_success = triggerAndProcessFLRoundInApi();

  if (api_success) {
    NS_LOG_INFO("StartNewFLRound: Python API call successful for round " << roundNumber);
    if (!selectedClientsForCurrentRound.empty()) {
      NS_LOG_INFO("StartNewFLRound: Scheduling ns-3 model uploads for "
                  << selectedClientsForCurrentRound.size()
                  << " clients with updated times/sizes.");
      sendModelsToServer();
      roundStartTimeNs3CommsParam = Simulator::Now();
      roundFinished = false;
      NS_LOG_INFO("StartNewFLRound: ns-3 comms phase started for round "
                  << roundNumber << " at "
                  << roundStartTimeNs3CommsParam.GetSeconds()
                  << "s. roundFinished set to false.");
    } else {
      NS_LOG_INFO("StartNewFLRound: Python API call successful, but no clients "
                  "selected by ns-3 for simulated upload. Marking round as "
                  "(comms-wise) finished.");
      roundFinished = true;
    }
  } else {
    NS_LOG_ERROR("StartNewFLRound: Python API call FAILED for round "
                 << roundNumber << ". Skipping ns-3 comms phase.");
    roundFinished = true;
  }
}

void FLCoordinator::manager() {
  static Time roundStartTimeNs3Comms = Simulator::Now();
  NS_LOG_INFO("Manager called at " << Simulator::Now().GetSeconds()
                                   << "s. RoundNumber: " << roundNumber
                                   << ", roundFinished (ns-3 comms): "
                                   << (roundFinished ? "true" : "false"));

  if (!fl_api_initialized) {
    NS_LOG_INFO("Manager: FL API not yet initialized by main. Manager waiting for 5 seconds.");
    Simulator::Schedule(Seconds(managerInterval), &FLCoordinator::manager);
    return;
  }

  if (!roundFinished) {
    NS_LOG_INFO("Manager: ns-3 communication phase for round " << roundNumber << " is ongoing.");
    if (isRoundTimedOut(roundStartTimeNs3Comms)) {
      NS_LOG_WARN("Manager: Round " << roundNumber << " ns-3 comms timed out.");
      logRoundTimeout();
      roundFinished = true;
    } else {
      nodesIPs = NetworkUtils::nodeToIps();
      bool all_selected_clients_finished_ns3_comms =
          NetworkUtils::checkFinishedTransmission(nodesIPs, selectedClientsForCurrentRound);

      if (all_selected_clients_finished_ns3_comms) {
        if (!selectedClientsForCurrentRound.empty() || !endOfStreamTimes.empty()) {
          NS_LOG_INFO("Manager: All selected clients ("
                      << endOfStreamTimes.size() << "/"
                      << selectedClientsForCurrentRound.size()
                      << ") completed ns-3 transmissions for round "
                      << roundNumber);
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
      finalizeNs3CommsPhase();
    }
  }

  if (roundFinished) {
    NS_LOG_INFO("Manager: ns-3 communication phase for round "
                << roundNumber << " is finished or was skipped.");

    startNewFLRound(roundStartTimeNs3Comms);
    MetricsCollector::exportDataFrames();
  }

  if (!Simulator::IsFinished()) {
    NS_LOG_INFO("Manager: Scheduling next call in " << managerInterval << " seconds.");
    Simulator::Schedule(Seconds(managerInterval), &FLCoordinator::manager);
  } else {
    NS_LOG_INFO("Manager: Simulation is finished, not scheduling next call.");
  }
}