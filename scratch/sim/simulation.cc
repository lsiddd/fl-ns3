// Filename: /home/lucas/fl-ns3/scratch/sim/simulation.cc
// Macro for logging - REPLACED WITH NS_LOG_COMPONENT_DEFINE
// #define LOG(x) std::cout << x << std::endl

// Project-specific headers
#include "MyApp.h"
#include "client_types.h"
#include "dataframe.h"
#include "notifications.h"
#include "utils.h"
#include "httplib.h" // ADICIONADO: Nossa biblioteca de cliente HTTP

// External library headers
#include "json.hpp"

// NS-3 module headers
#include "ns3/applications-module.h"
#include "ns3/command-line.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h" // Required for Simulator::Schedule
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-l3-protocol.h" // Include IPv4 headers for logging
#include "ns3/isotropic-antenna-model.h"
#include "ns3/log.h" // Include NS-3 log module
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/lte-ue-rrc.h" // Make sure LteUeRrc is included
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/tcp-socket-base.h" // Include TCP Socket Base headers for logging
#include "ns3/tcp-socket.h"      // Include TCP Socket headers for logging

// Standard Library headers
#include <algorithm>
#include <chrono>
#include <cmath> // For std::ceil
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip> // For std::fixed and std::setprecision
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h> // For sleep
#include <cstdio>
#include <stdexcept> 
#include <cstring>
#include <utility> // ADICIONADO: Para std::pair

// Using declarations for convenience
using namespace ns3;
using json = nlohmann::json;

// Define the simulation logging component
NS_LOG_COMPONENT_DEFINE("Simulation");

// Global constants
static constexpr double simStopTime = 400.0;
static constexpr int numberOfUes = 10;
static constexpr int numberOfEnbs = 5;
static constexpr int numberOfParticipatingClients = 5;
static constexpr int scenarioSize = 1000;
bool useStaticClients = true;
std::string algorithm = "fedavg";

// Helper function to get an environment variable or a default value
std::string getEnvVar(const std::string &key, const std::string &default_val) {
    const char* val = std::getenv(key.c_str());
    if (val == nullptr) {
        NS_LOG_INFO("Variavel de ambiente '" << key << "' nao encontrada. Usando valor padrao: '" << default_val << "'.");
        return default_val;
    }
    NS_LOG_INFO("Variavel de ambiente encontrada '" << key << "'. Valor: '" << val << "'.");
    return std::string(val);
}

// Read the API URL from the environment variable set by Docker Compose
std::string FL_API_BASE_URL = getEnvVar("FL_API_URL", "http://127.0.0.1:5005");
const int FL_API_NUM_CLIENTS = numberOfUes;
const int FL_API_CLIENTS_PER_ROUND =
    numberOfParticipatingClients;

DataFrame accuracy_df;
DataFrame participation_df;
DataFrame throughput_df;
DataFrame rsrp_sinr_df; 

// Global variables for simulation objects
NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;
NetDeviceContainer enbDevs;
NetDeviceContainer ueDevs;
Ipv4Address remoteHostAddr;

FlowMonitorHelper flowmon;

std::map<Ipv4Address, double> endOfStreamTimes;
std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;

// Global state variables
static bool roundFinished = true;
static int roundNumber = 0;
static bool fl_api_initialized = false;

// Client-related information
std::vector<NodesIps> nodesIPs;
std::vector<ClientModels>
    clientsInfoGlobal;
std::vector<ClientModels>
    selectedClientsForCurrentRound;

// Timeout for certain operations
Time timeout = Seconds(50);
static double constexpr managerInterval = 1.0; 

// REMOVIDO: A função 'exec' não é mais necessária.
// std::string exec(const char* cmd) { ... }


// --- MODIFICADO: Antiga função `callPythonApi` reimplementada com `httplib` ---
// Mantém a assinatura original para chamadas simples que não precisam do corpo da resposta.
int callPythonApi(const std::string &endpoint,
                  const std::string &method = "POST",
                  const std::string &data = "",
                  const std::string &output_file = "") { // output_file é ignorado
    
    std::string host;
    int port = -1;
    size_t host_start = FL_API_BASE_URL.find("://");
    host_start = (host_start == std::string::npos) ? 0 : host_start + 3;
    size_t port_start = FL_API_BASE_URL.find(":", host_start);

    if (port_start != std::string::npos) {
        host = FL_API_BASE_URL.substr(host_start, port_start - host_start);
        port = std::stoi(FL_API_BASE_URL.substr(port_start + 1));
    } else {
        host = FL_API_BASE_URL.substr(host_start);
        port = 80;
    }
    
    if (port == -1) {
        NS_LOG_ERROR("Nao foi possivel analisar a porta da URL_BASE_API_FL: " << FL_API_BASE_URL);
        return -1;
    }
    
    httplib::Client cli(host, port);
    cli.set_connection_timeout(5); 
    cli.set_read_timeout(60); // Timeout generoso para inicialização

    NS_LOG_INFO("callPythonApi: Preparando para executar " << method << " para o endpoint "
                << endpoint << " em " << Simulator::Now().GetSeconds() << "s.");

    httplib::Result res;
    if (method == "POST") {
        res = cli.Post(endpoint.c_str(), data, "application/json");
    } else { // Assume GET
        res = cli.Get(endpoint.c_str());
    }

    if (res) {
        if (res->status != 200) {
            NS_LOG_ERROR("callPythonApi: Erro na chamada da API para " << endpoint
                         << ". Status: " << res->status << ". Resposta: " << res->body);
        }
        return res->status;
    } else {
        auto err = res.error();
        NS_LOG_ERROR("callPythonApi: Falha na conexao da API para " << endpoint 
                     << ". Erro: " << httplib::to_string(err));
        return -1; // -1 indica falha de conexão
    }
}

// --- ADICIONADO: Nova função auxiliar para chamadas que precisam da resposta JSON ---
std::pair<int, json> callPythonApiForJson(const std::string &endpoint,
                                          const std::string &method,
                                          const std::string &data) {
    std::string host;
    int port = -1;
    size_t host_start = FL_API_BASE_URL.find("://");
    host_start = (host_start == std::string::npos) ? 0 : host_start + 3;
    size_t port_start = FL_API_BASE_URL.find(":", host_start);

    if (port_start != std::string::npos) {
        host = FL_API_BASE_URL.substr(host_start, port_start - host_start);
        port = std::stoi(FL_API_BASE_URL.substr(port_start + 1));
    } else {
        host = FL_API_BASE_URL.substr(host_start);
        port = 80;
    }
    
    if (port == -1) {
        NS_LOG_ERROR("Nao foi possivel analisar a porta da URL_BASE_API_FL: " << FL_API_BASE_URL);
        return {-1, {{"error", "URL invalida"}}};
    }

    httplib::Client cli(host, port);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(120); // As rodadas podem demorar

    httplib::Result res;
    if (method == "POST") {
        res = cli.Post(endpoint.c_str(), data, "application/json");
    } else {
        res = cli.Get(endpoint.c_str());
    }

    if (res) {
        json response_body;
        try {
            if (!res->body.empty()) {
                response_body = json::parse(res->body);
            }
        } catch (json::parse_error& e) {
            NS_LOG_ERROR("Falha ao analisar JSON da resposta de " << endpoint << ": " << e.what());
            return {res->status, {{"parse_error", e.what()}}};
        }
        return {res->status, response_body};
    } else {
        auto err = res.error();
        NS_LOG_ERROR("Falha na conexao da API para " << endpoint << ". Erro: " << httplib::to_string(err));
        return {-1, {{"connection_error", httplib::to_string(err)}}};
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
  std::vector<std::string> throughput_columns = {
      "time", "tx_throughput_mbps", "rx_throughput_mbps", "total_tx_bytes",
      "total_rx_bytes"};
  std::vector<std::string> rsrp_sinr_columns = {
      "time",    "round",    "ue_node_id", "enb_cell_id",
      "ue_rnti", "rsrp_dbm", "sinr_db",    "connected_state"};

  for (const auto &column : accuracy_columns) {
    accuracy_df.addColumn(column);
    NS_LOG_DEBUG("Adicionada coluna accuracy_df: " << column);
  }
  for (const auto &column : participation_columns) {
    participation_df.addColumn(column);
    NS_LOG_DEBUG("Adicionada coluna participation_df: " << column);
  }
  for (const auto &column : throughput_columns) {
    throughput_df.addColumn(column);
    NS_LOG_DEBUG("Adicionada coluna throughput_df: " << column);
  }
  for (const auto &column : rsrp_sinr_columns) {
    rsrp_sinr_df.addColumn(column);
    NS_LOG_DEBUG("Adicionada coluna rsrp_sinr_df: " << column);
  }
  NS_LOG_INFO("Todos os DataFrames inicializados com colunas.");
}

std::pair<double, double> getRsrpSinr(uint32_t nodeIdx) {
  Ptr<NetDevice> ueDevice = ueDevs.Get(nodeIdx);
  if (!ueDevice) {
    NS_LOG_DEBUG("getRsrpSinr: Dispositivo UE no indice " << nodeIdx << " e nulo.");
    return {0.0, 0.0};
  }
  auto lteUeNetDevice = ueDevice->GetObject<LteUeNetDevice>();
  if (!lteUeNetDevice) {
    NS_LOG_DEBUG("getRsrpSinr: NetDevice no indice "
                 << nodeIdx << " nao e um LteUeNetDevice.");
    return {0.0, 0.0};
  }
  auto rrc = lteUeNetDevice->GetRrc();

  std::string connected_state = "NOT_CONNECTED";
  if (!rrc || (rrc->GetState() != LteUeRrc::CONNECTED_NORMALLY &&
               rrc->GetState() != LteUeRrc::CONNECTED_HANDOVER)) {
    
    NS_LOG_DEBUG("getRsrpSinr: UE Node "
                 << ueNodes.Get(nodeIdx)->GetId()
                 << " RRC nao esta em estado conectado. Estado: "
                 << (rrc ? rrc->GetState() : LteUeRrc::IDLE_START));
    rsrp_sinr_df.addRow({Simulator::Now().GetSeconds(), roundNumber,
                         ueNodes.Get(nodeIdx)->GetId(), (uint32_t)0,
                         (uint32_t)0, 0.0, 0.0, connected_state});
    return {0.0, 0.0};
  }

  connected_state =
      (rrc->GetState() == LteUeRrc::CONNECTED_NORMALLY ? "CONNECTED_NORMALLY"
                                                       : "CONNECTED_HANDOVER");
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

  rsrp_sinr_df.addRow({Simulator::Now().GetSeconds(), roundNumber,
                       ueNodes.Get(nodeIdx)->GetId(), (uint32_t)cellId,
                       (uint32_t)rnti, rsrp, sinr, connected_state});
  NS_LOG_DEBUG("getRsrpSinr: UE Node "
               << ueNodes.Get(nodeIdx)->GetId() << " (CellId: " << cellId
               << ", RNTI: " << rnti << ") RSRP: " << rsrp
               << " dBm, SINR: " << sinr << " dB. Estado: " << connected_state);
  return {rsrp, sinr};
}

void updateAllClientsGlobalInfo() {
  NS_LOG_INFO("Atualizando informacoes globais de clientes para todos os UEs.");
  clientsInfoGlobal.clear();
  const int defaultTrainingTime =
      5005;
  const int defaultModelSizeBytes =
      2000000;

  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    auto [rsrp, sinr] = getRsrpSinr(i);
    double placeholderAccuracy = 0.1;
    clientsInfoGlobal.emplace_back(ueNodes.Get(i), defaultTrainingTime,
                                   defaultModelSizeBytes, rsrp, sinr,
                                   placeholderAccuracy);
    NS_LOG_DEBUG("  UE Node " << ueNodes.Get(i)->GetId() << ": RSRP=" << rsrp
                              << " dBm, SINR=" << sinr << " dB.");
  }
  NS_LOG_INFO("Informacoes globais de clientes atualizadas para "
              << clientsInfoGlobal.size() << " UEs.");
}

void selectNs3ManagedClients(int n_to_select) {
  NS_LOG_INFO("Selecionando " << n_to_select << " clientes para a rodada de FL "
                           << roundNumber << " com base nos criterios do ns-3.");
  selectedClientsForCurrentRound.clear();
  updateAllClientsGlobalInfo(); 

  std::vector<ClientModels> candidates = clientsInfoGlobal;
  std::sort(candidates.begin(), candidates.end(),
            [](const ClientModels &a, const ClientModels &b) {
              return a.sinr > b.sinr;
            });

  int actual_selected_count = 0;
  for (int i = 0; i < n_to_select && (long unsigned int)i < candidates.size();
       ++i) {
    if (candidates[i].sinr > 0.001 ||
        candidates[i].rsrp < 0.0) {
      ClientModels selected_client = candidates[i];
      selected_client.selected =
          true;
      selectedClientsForCurrentRound.push_back(selected_client);
      actual_selected_count++;
      NS_LOG_DEBUG("  Cliente selecionado " << selected_client.node->GetId()
                                        << " (SINR: " << selected_client.sinr
                                        << " dB, RSRP: " << selected_client.rsrp
                                        << " dBm)");
    } else {
      NS_LOG_DEBUG("  Ignorando cliente "
                   << candidates[i].node->GetId() << " devido a baixo SINR ("
                   << candidates[i].sinr << " dB) ou RSRP ("
                   << candidates[i].rsrp << " dBm).");
    }
  }
  NS_LOG_INFO("ns-3 selecionou " << actual_selected_count << " clientes (de "
                               << n_to_select << " solicitados) para a rodada de FL "
                               << roundNumber);
  if (actual_selected_count == 0 && n_to_select > 0) {
    NS_LOG_WARN("Nenhum cliente elegivel foi selecionado pelo ns-3 para esta rodada, "
                "possivelmente devido a mas condicoes de rede para todos os UEs.");
  }
}

// MODIFICADO: Esta função agora usa a nova função auxiliar e manipula o JSON diretamente.
bool triggerAndProcessFLRoundInApi() {
  NS_LOG_INFO("=================== Acionando Rodada de FL "
              << roundNumber << " na API Python em "
              << Simulator::Now().GetSeconds() << "s ===================");

  if (selectedClientsForCurrentRound.empty() &&
      FL_API_CLIENTS_PER_ROUND >
          0)
  {
    NS_LOG_INFO(
        "Nenhum cliente foi selecionado pelo ns-3 para esta rodada. Ignorando chamada de API "
        "para /run_round.");
    if (FL_API_CLIENTS_PER_ROUND > 0)
      return false;
  }

  json client_indices_payload;
  std::vector<int> indices_list;
  for (const auto &client : selectedClientsForCurrentRound) {
    indices_list.push_back(static_cast<int>(client.node->GetId()));
  }
  client_indices_payload["client_indices"] = indices_list;

  NS_LOG_INFO("Chamando /run_round para a rodada "
              << roundNumber << " com payload (primeiros 100 caracteres): "
              << client_indices_payload.dump().substr(0, 100) << "...");
  
  auto [http_code, response_json] = callPythonApiForJson("/run_round", "POST", client_indices_payload.dump());

  if (http_code == 200) {
    NS_LOG_INFO("Chamada a API Python /run_round bem-sucedida para a rodada " << roundNumber);
    NS_LOG_INFO("Resposta da API Python (primeiros 200 caracteres): "
                << response_json.dump(2).substr(0, 200) << "...");
    
    accuracy_df.addRow(
        {Simulator::Now().GetSeconds(), roundNumber,
         response_json.value("global_test_accuracy", 0.0),
         response_json.value("global_test_loss", 0.0),
         response_json.value("avg_client_accuracy", 0.0),
         response_json.value("avg_client_loss", 0.0),
         response_json.value("round_duration_seconds", 0.0)});
    NS_LOG_INFO("Dados de acuracia adicionados ao DataFrame para a rodada " << roundNumber);

    auto client_perf_details =
        response_json.value("simulated_client_performance", json::object());
    NS_LOG_INFO("Atualizando informacoes dos clientes selecionados com valores simulados da resposta da API ("
                << client_perf_details.size() << " clientes)...");
    for (auto const &[client_id_str, perf_data] :
         client_perf_details.items()) {
      try {
        int client_id = std::stoi(client_id_str);
        for (auto &client_model_info : selectedClientsForCurrentRound) {
          if (client_model_info.node->GetId() == (uint32_t)client_id) {
            client_model_info.nodeTrainingTime =
                perf_data.value("training_time_ms",
                                client_model_info.nodeTrainingTime);
            client_model_info.nodeModelSize =
                perf_data.value("model_size_bytes",
                                client_model_info.nodeModelSize);
            NS_LOG_DEBUG("  Cliente atualizado "
                         << client_id << ": training_time="
                         << client_model_info.nodeTrainingTime
                         << "ms, model_size="
                         << client_model_info.nodeModelSize << " bytes.");
            break; 
          }
        }
      } catch (const std::invalid_argument &ia) {
        NS_LOG_ERROR(
            "  Falha ao analisar a string de ID do cliente: " << client_id_str);
      }
    }
    return true;
  } else {
    NS_LOG_ERROR(
        "ERRO: Chamada a API Python /run_round falhou. Codigo HTTP: " << http_code);
    NS_LOG_ERROR("Resposta de Erro da API Python: " << response_json.dump(2));
    return false;
  }
}

void sendModelsToServer() { 
  NS_LOG_INFO("ns-3: Simulando uploads de modelo para "
              << selectedClientsForCurrentRound.size() << " clientes selecionados.");
  if (selectedClientsForCurrentRound.empty()) {
    NS_LOG_INFO("  Nenhum cliente selecionado para upload de modelo no ns-3 nesta rodada. "
                "Ignorando envio.");
    return;
  }

  for (const auto &client_model_info : selectedClientsForCurrentRound) {
    NS_LOG_INFO("  Cliente " << client_model_info.node->GetId()
                            << " agendando envio de modelo no ns-3 de tamanho "
                            << client_model_info.nodeModelSize << " bytes "
                            << "apos " << client_model_info.nodeTrainingTime
                            << "ms de tempo de pseudo-treinamento.");

    Simulator::Schedule(MilliSeconds(client_model_info.nodeTrainingTime),
                        &sendStream, client_model_info.node,
                        remoteHostContainer.Get(0),
                        client_model_info.nodeModelSize);
  }
}

bool isRoundTimedOut(Time roundStartTimeNs3Comms) {
  bool timedOut = Simulator::Now() - roundStartTimeNs3Comms > timeout;
  if (timedOut) {
    NS_LOG_WARN("isRoundTimedOut: Fase de Comunicacoes do ns-3 para a rodada "
                << roundNumber << " expirou em "
                << Simulator::Now().GetSeconds() << "s.");
  } else {
    NS_LOG_DEBUG("isRoundTimedOut: Fase de Comunicacoes do ns-3 para a rodada "
                 << roundNumber << " ainda nao expirou. Duracao atual: "
                 << (Simulator::Now() - roundStartTimeNs3Comms).GetSeconds()
                 << "s.");
  }
  return timedOut;
}

void logRoundTimeout() {
  NS_LOG_WARN("Rodada de Comunicacoes do ns-3 expirou para a rodada "
              << roundNumber
              << ". Transferencias bem-sucedidas: " << endOfStreamTimes.size() << "/"
              << selectedClientsForCurrentRound.size() << " clientes.");
}

void addParticipationToDataFrame() {
  NS_LOG_INFO("Adicionando dados de participacao ao DataFrame para a rodada " << roundNumber
                                                                  << ".");
  participation_df.addRow({Simulator::Now().GetSeconds(), roundNumber,
                           (uint32_t)selectedClientsForCurrentRound.size(),
                           (uint32_t)endOfStreamTimes.size()});
  NS_LOG_INFO("  Registrado selecionados: " << selectedClientsForCurrentRound.size()
                                      << ", comunicacoes ns-3 concluidas: "
                                      << endOfStreamTimes.size());
}

void finalizeNs3CommsPhase() {
  NS_LOG_INFO("Fase de Comunicacoes do ns-3 para a rodada "
              << roundNumber << " finalizada em " << Simulator::Now().GetSeconds()
              << ". Transferencias bem-sucedidas: " << endOfStreamTimes.size() << "/"
              << selectedClientsForCurrentRound.size());

  addParticipationToDataFrame();
  roundCleanup(); 
  NS_LOG_INFO("Limpeza da fase de comunicacoes do ns-3 concluida para a rodada " << roundNumber
                                                             << ".");
}

void startNewFLRound(
    Time &roundStartTimeNs3CommsParam) 
{
  roundNumber++;
  NS_LOG_INFO("StartNewFLRound: Iniciando para a Rodada de FL "
              << roundNumber << " em " << Simulator::Now().GetSeconds()
              << "s.");

  selectNs3ManagedClients(FL_API_CLIENTS_PER_ROUND);
  NS_LOG_INFO("StartNewFLRound: ns-3 selecionou "
              << selectedClientsForCurrentRound.size()
              << " clientes para esta rodada.");

  if (selectedClientsForCurrentRound.empty() && FL_API_CLIENTS_PER_ROUND > 0) {
    NS_LOG_INFO(
        "StartNewFLRound: Nenhum cliente foi selecionado pelo ns-3 (e.g., devido a SINR "
        "ou outros criterios, ou nenhum UE elegivel). Ignorando chamada a API e comunicacoes ns-3 "
        "para esta rodada.");
    roundFinished = true;
    return;
  }

  NS_LOG_INFO("StartNewFLRound: Acionando rodada de FL na API Python para a rodada "
              << roundNumber);
  bool api_success = triggerAndProcessFLRoundInApi();

  if (api_success) {
    NS_LOG_INFO("StartNewFLRound: Chamada a API Python bem-sucedida para a rodada "
                << roundNumber);
    if (!selectedClientsForCurrentRound.empty()) {
      NS_LOG_INFO("StartNewFLRound: Agendando uploads de modelo no ns-3 para "
                  << selectedClientsForCurrentRound.size()
                  << " clientes com tempos/tamanhos atualizados.");
      sendModelsToServer();
      roundStartTimeNs3CommsParam =
          Simulator::Now(); 
      roundFinished = false; 
      NS_LOG_INFO("StartNewFLRound: fase de comunicacoes do ns-3 iniciada para a rodada "
                  << roundNumber << " em "
                  << roundStartTimeNs3CommsParam.GetSeconds()
                  << "s. roundFinished definido como false.");
    } else {
      NS_LOG_INFO("StartNewFLRound: Chamada a API Python bem-sucedida, mas nenhum cliente "
                  "selecionado pelo ns-3 para upload simulado. Marcando rodada como "
                  "(em termos de comunicacao) finalizada.");
      roundFinished = true;
    }
  } else {
    NS_LOG_ERROR("StartNewFLRound: Chamada a API Python FALHOU para a rodada "
                 << roundNumber << ". Ignorando fase de comunicacoes ns-3.");
    roundFinished = true;
  }
}
void exportDataFrames() {
    NS_LOG_INFO("Exportando DataFrames para o diretorio 'results/'.");
    system("mkdir -p results");
    accuracy_df.toCsv("results/accuracy_fl_api.csv");
    participation_df.toCsv("results/clientParticipation_fl_api.csv");
    throughput_df.toCsv("results/throughput_fl_api.csv");
    rsrp_sinr_df.toCsv("results/rsrp_sinr_metrics.csv");
    NS_LOG_INFO("Todos os DataFrames exportados.");
}

void manager() {
  static Time roundStartTimeNs3Comms =
      Simulator::Now(); 
  NS_LOG_INFO("Manager chamado em " << Simulator::Now().GetSeconds()
                                   << "s. RoundNumber: " << roundNumber
                                   << ", roundFinished (ns-3 comms): "
                                   << (roundFinished ? "true" : "false"));

  if (!fl_api_initialized) {
    NS_LOG_INFO("Manager: API de FL ainda nao inicializada pelo main. Manager aguardando "
                "por 5 segundos.");
    Simulator::Schedule(Seconds(managerInterval), &manager); 
    return;
  }

  if (!roundFinished) {
    NS_LOG_INFO("Manager: Fase de comunicacao do ns-3 para a rodada "
                << roundNumber << " esta em andamento.");
    if (isRoundTimedOut(roundStartTimeNs3Comms)) {
      NS_LOG_WARN("Manager: Rodada " << roundNumber << " ns-3 comms expirou.");
      logRoundTimeout(); 
      roundFinished = true;
    } else {
      nodesIPs = nodeToIps(); 
      bool all_selected_clients_finished_ns3_comms =
          checkFinishedTransmission(nodesIPs, selectedClientsForCurrentRound);

      if (all_selected_clients_finished_ns3_comms) {
        if (!selectedClientsForCurrentRound.empty() ||
            !endOfStreamTimes
                 .empty()) { 
          NS_LOG_INFO("Manager: Todos os clientes selecionados ("
                      << endOfStreamTimes.size() << "/"
                      << selectedClientsForCurrentRound.size()
                      << ") completaram as transmissoes ns-3 para a rodada "
                      << roundNumber);
        } else if (selectedClientsForCurrentRound.empty()) {
          NS_LOG_INFO(
              "Manager: Nenhum cliente foi selecionado para comunicacoes ns-3 na rodada "
              << roundNumber << ", considerando a fase de comunicacoes completa.");
        }
        roundFinished = true;
      } else {
        NS_LOG_INFO(
            "Manager: Aguardando por "
            << selectedClientsForCurrentRound.size() - endOfStreamTimes.size()
            << " mais clientes para finalizar as comunicacoes ns-3 para a rodada " << roundNumber);
      }
    }

    if (roundFinished) {
      NS_LOG_INFO("Manager: Finalizando a fase de comunicacoes do ns-3 para a rodada "
                  << roundNumber);
      finalizeNs3CommsPhase(); 
    }
  }

  if (roundFinished) {
    NS_LOG_INFO("Manager: Fase de comunicacao do ns-3 para a rodada "
                << roundNumber << " esta finalizada ou foi ignorada.");

    startNewFLRound(roundStartTimeNs3Comms);
    exportDataFrames();
  }

  if (!Simulator::IsFinished()) {
    NS_LOG_INFO("Manager: Agendando proxima chamada em " << managerInterval
                                                    << " segundos.");
    Simulator::Schedule(Seconds(managerInterval), &manager);
  } else {
    NS_LOG_INFO("Manager: Simulacao finalizada, nao agendando proxima chamada.");
  }
}

void ConfigureDefaults() {
  const uint32_t maxTxBufferSizeUm = 10 * 1024 * 1024 * 10;
  const uint32_t maxTxBufferSizeAm = 10 * 1024 * 1024;
  const uint32_t maxTxBufferSizeLowLat = 10 * 1024 * 1024;

  Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize",
                     UintegerValue(maxTxBufferSizeUm));
  Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize",
                     UintegerValue(maxTxBufferSizeAm));
  Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize",
                     UintegerValue(maxTxBufferSizeLowLat));
  Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                     TypeIdValue(TcpCubic::GetTypeId()));
  Config::SetDefault("ns3::TcpSocketBase::MinRto",
                     TimeValue(MilliSeconds(200)));
  Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout",
                     TimeValue(Seconds(2)));
  Config::SetDefault("ns3::TcpSocket::SegmentSize",
                     UintegerValue(1448)); 
  Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
  uint32_t sndRcvBufSize = 131072 * 10;
  Config::SetDefault("ns3::TcpSocket::SndBufSize",
                     UintegerValue(sndRcvBufSize));
  Config::SetDefault("ns3::TcpSocket::RcvBufSize",
                     UintegerValue(sndRcvBufSize));
  Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));
  Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(20.0));
  NS_LOG_INFO("Configuracoes padrao do NS-3 aplicadas.");
}

// Main function
int main(int argc, char *argv[]) {
  LogComponentEnable("Simulation", LOG_LEVEL_INFO);
  // LogComponentEnable("MyApp", LOG_LEVEL_DEBUG);
  LogComponentEnable("Utils", LOG_LEVEL_INFO);
  LogComponentEnable("ClientTypes", LOG_LEVEL_INFO);
  LogComponentEnable("DataFrame", LOG_LEVEL_DEBUG); 
  LogComponentEnable("Notifications",LOG_LEVEL_INFO); 
  // LogComponentEnable("TcpSocket", LOG_LEVEL_DEBUG); 
  // LogComponentEnable("TcpSocketBase", LOG_LEVEL_DEBUG);

  ConfigureDefaults();
  initializeDataFrames();

  CommandLine cmd;
  cmd.AddValue("algorithm",
               "FL algorithm (ns-3 perspective, less relevant now)", algorithm);
  cmd.Parse(argc, argv);


  // --- MODIFICADO: Chamadas de configuração agora usam a função `callPythonApi` reimplementada ---
  NS_LOG_INFO("Configurando a API Python de FL...");
  int api_port = 5005;
  json fl_config_payload;
  fl_config_payload["dataset"] = "mnist";
  fl_config_payload["num_clients"] = FL_API_NUM_CLIENTS;
  fl_config_payload["clients_per_round"] =
      FL_API_CLIENTS_PER_ROUND;
  fl_config_payload["local_epochs"] = 1;
  fl_config_payload["batch_size"] = 32;
  fl_config_payload["port"] = api_port;

  int http_code =
      callPythonApi("/configure", "POST",
                    fl_config_payload.dump());
  if (http_code != 200) {
    NS_LOG_ERROR(
        "ERRO: Falha ao configurar a API Python de FL. Codigo HTTP: " << http_code);
  } else {
    NS_LOG_INFO("API Python de FL configurada com sucesso.");
  }
  sleep(1); 

  NS_LOG_INFO(
      "Inicializando a simulacao da API Python de FL (carregamento de dados, modelo inicial)...");
  http_code = callPythonApi("/initialize_simulation",
                            "POST");
  if (http_code != 200) {
    NS_LOG_ERROR(
        "ERRO: Falha ao inicializar a simulacao da API Python de FL. Codigo HTTP: "
        << http_code);
  } else {
    NS_LOG_INFO("Simulacao da API Python de FL inicializada com sucesso.");
    fl_api_initialized = true;
  }
  sleep(1); 

  // --- ns-3 Network Setup (sem alterações) ---
  Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
  mmwaveHelper->SetEpcHelper(epcHelper);
  mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
  mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
  NS_LOG_INFO("LTE Helper e EPC Helper criados e configurados.");

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults();

  Ptr<Node> pgw = epcHelper->GetPgwNode();
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  InternetStackHelper internet;
  internet.Install(remoteHostContainer);
  NS_LOG_INFO(
      "PGW e RemoteHost criados e InternetStack instalado no RemoteHost.");

  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
  p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
  p2ph.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
  remoteHostAddr = internetIpIfaces.GetAddress(1);
  NS_LOG_INFO("Link Ponto-a-Ponto entre PGW e RemoteHost configurado. "
              "IP do RemoteHost: "
              << remoteHostAddr);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
      ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                             Ipv4Mask("255.0.0.0"), 1);
  NS_LOG_INFO("Rota estatica adicionada no RemoteHost para a rede UE.");

  enbNodes.Create(numberOfEnbs);
  ueNodes.Create(numberOfUes);
  NS_LOG_INFO("Criados " << numberOfEnbs << " eNBs e " << numberOfUes
                         << " UEs.");

  MobilityHelper enbmobility;
  Ptr<RandomRectanglePositionAllocator> enbPositionAlloc =
      CreateObject<RandomRectanglePositionAllocator>();
  std::string enbBounds = "0|" + std::to_string(scenarioSize) + "|0|" +
                          std::to_string(scenarioSize);
  enbPositionAlloc->SetAttribute(
      "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                       std::to_string(scenarioSize) + "]"));
  enbPositionAlloc->SetAttribute(
      "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                       std::to_string(scenarioSize) + "]"));
  enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  enbmobility.SetPositionAllocator(enbPositionAlloc);
  enbmobility.Install(enbNodes);
  NS_LOG_INFO("eNBs instalados com ConstantPositionMobilityModel e posicoes aleatorias dentro do tamanho do cenario.");

  MobilityHelper uemobility;
  if (useStaticClients) {
    NS_LOG_INFO("Instalando ConstantPositionMobilityModel para UEs estaticos com posicoes aleatorias.");
    uemobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<RandomRectanglePositionAllocator> uePositionAlloc =
        CreateObject<RandomRectanglePositionAllocator>();
    std::string ueBounds = "0|" + std::to_string(scenarioSize) + "|0|" +
                           std::to_string(scenarioSize);
    uePositionAlloc->SetAttribute(
        "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                         std::to_string(scenarioSize) + "]"));
    uePositionAlloc->SetAttribute(
        "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                         std::to_string(scenarioSize) + "]"));

    uemobility.SetPositionAllocator(uePositionAlloc);
    uemobility.Install(ueNodes);
    NS_LOG_INFO("UEs estaticos instalados com ConstantPositionMobilityModel e posicoes aleatorias dentro do tamanho do cenario.");
  } else {
    NS_LOG_INFO("Instalando RandomWalk2dMobilityModel para UEs moveis.");
    std::string walkBounds = "0|" + std::to_string(scenarioSize) + "|0|" +
                             std::to_string(scenarioSize);
    uemobility.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel", "Mode", StringValue("Time"), "Time",
        StringValue("2s"), "Speed",
        StringValue("ns3::ConstantRandomVariable[Constant=20.0]"),
        "Bounds", StringValue(walkBounds));
    uemobility.Install(ueNodes);
    NS_LOG_INFO("UEs moveis instalados com RandomWalk2dMobilityModel dentro dos limites do tamanho do cenario.");
  }

  enbmobility.Install(
      pgw);
  enbmobility.Install(remoteHost); 
  NS_LOG_INFO("Modelos de mobilidade instalados no PGW e RemoteHost para o NetAnim.");

  enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
  ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);
  internet.Install(ueNodes);
  epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
  mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs); 
  NS_LOG_INFO("Dispositivos eNB e UE instalados. Enderecos IP dos UEs atribuidos. UEs "
              "conectados ao eNB mais proximo.");

  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<Node> ueNode = ueNodes.Get(i);
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(),
                                     1);
  }
  NS_LOG_INFO("Rotas estaticas definidas para os UEs.");

  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();
    uePhy->TraceConnectWithoutContext(
        "ReportCurrentCellRsrpSinr",
        MakeCallback<void, uint16_t, uint16_t, double, double, uint8_t>(
            &ReportUeSinrRsrp));
  }
  NS_LOG_INFO("Fontes de trace RSRP/SINR conectadas para os UEs.");

  Ptr<FlowMonitor> monitor = flowmon.InstallAll();
  NS_LOG_INFO("FlowMonitor instalado.");

  Simulator::Schedule(Seconds(2.0),
                      &manager);
  Simulator::Schedule(Seconds(1.0), &networkInfo,
                      monitor);
  NS_LOG_INFO("Funcoes Manager e networkInfo agendadas.");

  AnimationInterface anim("fl_api_mmwave_animation.xml");
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
  NS_LOG_INFO("Configuracao do NetAnim completa.");

  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedEnb));
  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedUe));
  NS_LOG_INFO("Fontes de trace LTE ConnectionEstablished conectadas.");

  Simulator::Stop(Seconds(simStopTime));
  NS_LOG_INFO("Iniciando a Simulacao ns-3. A simulacao parara em "
              << simStopTime << "s.");
  Simulator::Run();
  NS_LOG_INFO("Simulacao ns-3 Finalizada.");

  NS_LOG_INFO("Parando o servidor da API Python de FL...");
  int kill_status = system("pkill -f 'python3 scratch/api.py'");
  if (kill_status == -1) {
    NS_LOG_ERROR("Falha ao executar o comando pkill: " << std::strerror(errno));
  } else {
    if (WIFEXITED(kill_status)) {
      const int exit_code = WEXITSTATUS(kill_status);
      if (exit_code == 0) {
        NS_LOG_INFO("Servidor da API Python de FL terminado com sucesso");
      } else if (exit_code == 1) {
        NS_LOG_WARN("Servidor Python nao encontrado (ja terminado?)");
      } else {
        NS_LOG_WARN("pkill saiu de forma anormal (codigo: " << exit_code << ")");
      }
    } else {
      NS_LOG_WARN("pkill terminado por sinal: " << WTERMSIG(kill_status));
    }
  }

  exportDataFrames();
  Simulator::Destroy();
  NS_LOG_INFO("Simulador Destruido.");
  return 0;
}