#include "metrics_collector.h"
#include "client_types.h"
#include "network_setup.h"
#include "ns3/log.h"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cstdlib>

NS_LOG_COMPONENT_DEFINE("MetricsCollector");

DataFrame accuracy_df;
DataFrame participation_df;
DataFrame throughput_df;
DataFrame rsrp_sinr_df;
DataFrame fl_metrics_df;
std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;

extern std::vector<ClientModels> clientsInfoGlobal;
extern int roundNumber;

void MetricsCollector::initializeDataFrames() {
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
  std::vector<std::string> fl_metrics_columns = {
      "time", "round", "client_id", "accuracy"};

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
  for (const auto &column : fl_metrics_columns) {
    fl_metrics_df.addColumn(column);
    NS_LOG_DEBUG("Added fl_metrics_df column: " << column);
  }
  NS_LOG_INFO("All DataFrames initialized with columns.");
}

std::pair<double, double> MetricsCollector::getRsrpSinr(uint32_t nodeIdx) {
  Ptr<NetDevice> ueDevice = ueDevs.Get(nodeIdx);
  if (!ueDevice) {
    NS_LOG_DEBUG("getRsrpSinr: UE device at index " << nodeIdx << " is null.");
    return {0.0, 0.0};
  }
  auto lteUeNetDevice = ueDevice->GetObject<LteUeNetDevice>();
  if (!lteUeNetDevice) {
    NS_LOG_DEBUG("getRsrpSinr: NetDevice at index "
                 << nodeIdx << " is not an LteUeNetDevice.");
    return {0.0, 0.0};
  }
  auto rrc = lteUeNetDevice->GetRrc();

  std::string connected_state = "NOT_CONNECTED";
  if (!rrc || (rrc->GetState() != LteUeRrc::CONNECTED_NORMALLY &&
               rrc->GetState() != LteUeRrc::CONNECTED_HANDOVER)) {
    NS_LOG_DEBUG("getRsrpSinr: UE Node "
                 << ueNodes.Get(nodeIdx)->GetId()
                 << " RRC not in connected state. State: "
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
               << " dBm, SINR: " << sinr << " dB. State: " << connected_state);
  return {rsrp, sinr};
}

void MetricsCollector::updateAllClientsGlobalInfo() {
  NS_LOG_INFO("Updating global client information for all UEs.");
  clientsInfoGlobal.clear();
  const int defaultTrainingTime = 5000;
  const int defaultModelSizeBytes = 2000000;

  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    auto [rsrp, sinr] = getRsrpSinr(i);
    double placeholderAccuracy = 0.1;
    clientsInfoGlobal.emplace_back(ueNodes.Get(i), defaultTrainingTime,
                                   defaultModelSizeBytes, rsrp, sinr,
                                   placeholderAccuracy);
    NS_LOG_DEBUG("  UE Node " << ueNodes.Get(i)->GetId() << ": RSRP=" << rsrp
                              << " dBm, SINR=" << sinr << " dB.");
  }
  NS_LOG_INFO("Global client information updated for "
              << clientsInfoGlobal.size() << " UEs.");
}

void MetricsCollector::reportUeSinrRsrp(uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId) {
    sinrUe[cellId][rnti] = sinr;
    rsrpUe[cellId][rnti] = rsrp;
    NS_LOG_DEBUG("ReportUeSinrRsrp: Stored SINR=" << sinr << " and RSRP=" << rsrp << " for CellID=" << cellId << ", RNTI=" << rnti);
}

void MetricsCollector::reportUeSinrRsrp(std::string context,
                      uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId) {
    NS_LOG_DEBUG("ReportUeSinrRsrp (context version) - Context: '" << context
                                                                   << "', CellID: " << cellId << ", RNTI: " << rnti
                                                                   << ", RSRP: " << std::fixed << std::setprecision(2) << rsrp << " dBm"
                                                                   << ", SINR: " << std::fixed << std::setprecision(2) << sinr << " dB"
                                                                   << ", CC ID: " << (unsigned int)componentCarrierId);

    NS_LOG_DEBUG("ReportUeSinrRsrp: Calling non-context version of ReportUeSinrRsrp.");
    reportUeSinrRsrp(cellId, rnti, rsrp, sinr, componentCarrierId);
    NS_LOG_DEBUG("ReportUeSinrRsrp: Returned from non-context version of ReportUeSinrRsrp.");
}

void MetricsCollector::setupRsrpSinrTracing() {
  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();
    uePhy->TraceConnectWithoutContext(
        "ReportCurrentCellRsrpSinr",
        MakeCallback<void, uint16_t, uint16_t, double, double, uint8_t>(
            &ReportUeSinrRsrp));
  }
  NS_LOG_INFO("RSRP/SINR trace sources connected for UEs.");
}

void MetricsCollector::networkInfo(Ptr<FlowMonitor> monitor) {
    static Time lastTime = Seconds(0);
    static uint64_t lastTotalRxBytes = 0;
    static uint64_t lastTotalTxBytes = 0;

    NS_LOG_DEBUG("networkInfo: Scheduled at " << Simulator::Now().GetSeconds() << "s. Last time: " << lastTime.GetSeconds()
                                              << "s, lastTotalRxBytes: " << lastTotalRxBytes << ", lastTotalTxBytes: " << lastTotalTxBytes);

    Simulator::Schedule(Seconds(1.0), &MetricsCollector::networkInfo, monitor);
    NS_LOG_DEBUG("networkInfo: Scheduled next call for 1.0s from now.");

    if (!monitor) {
        NS_LOG_ERROR("networkInfo: FlowMonitor Ptr is null. Cannot gather stats. Skipping this interval.");
        return;
    }

    monitor->CheckForLostPackets();
    NS_LOG_DEBUG("networkInfo: Called CheckForLostPackets().");
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    NS_LOG_DEBUG("networkInfo: Retrieved " << stats.size() << " flow stats entries.");

    uint64_t currentTotalRxBytes = 0;
    uint64_t currentTotalTxBytes = 0;

    for (auto i = stats.begin(); i != stats.end(); ++i) {
        NS_LOG_DEBUG("Flow ID: " << i->first << ", RxBytes: " << i->second.rxBytes << ", TxBytes: " << i->second.txBytes
                                 << ", Packets Rx: " << i->second.rxPackets << ", Packets Tx: " << i->second.txPackets
                                 << ", Lost: " << i->second.lostPackets << ", Jitter: " << i->second.jitterSum.GetSeconds());
        currentTotalRxBytes += i->second.rxBytes;
        currentTotalTxBytes += i->second.txBytes;
    }
    NS_LOG_DEBUG("networkInfo: Current total Rx Bytes sum: " << currentTotalRxBytes << ", Current total Tx Bytes sum: " << currentTotalTxBytes);

    Time currentTime = Simulator::Now();
    double timeDiff = (currentTime - lastTime).GetSeconds();
    NS_LOG_DEBUG("networkInfo: Current time: " << currentTime.GetSeconds() << "s. Time difference: " << timeDiff << "s.");

    if (timeDiff > 0) {
        double instantRxThroughputMbps = static_cast<double>(currentTotalRxBytes - lastTotalRxBytes) * 8.0 / timeDiff / 1e6;
        double instantTxThroughputMbps = static_cast<double>(currentTotalTxBytes - lastTotalTxBytes) * 8.0 / timeDiff / 1e6;

        NS_LOG_INFO(currentTime.GetSeconds() << "s: Instant Rx Throughput: " << std::fixed << std::setprecision(4) << instantRxThroughputMbps << " Mbps, "
                                             << "Instant Tx Throughput: " << std::fixed << std::setprecision(4) << instantTxThroughputMbps << " Mbps.");

        throughput_df.addRow({currentTime.GetSeconds(), instantTxThroughputMbps, instantRxThroughputMbps, (uint32_t)currentTotalTxBytes, (uint32_t)currentTotalRxBytes});
        NS_LOG_DEBUG("networkInfo: Added row to throughput_df: Time=" << currentTime.GetSeconds()
                                                                      << ", TxThroughput=" << instantTxThroughputMbps << ", RxThroughput=" << instantRxThroughputMbps
                                                                      << ", TotalTxBytes=" << currentTotalTxBytes << ", TotalRxBytes=" << currentTotalRxBytes);
    } else if (currentTime == lastTime && (currentTotalRxBytes != lastTotalRxBytes || currentTotalTxBytes != lastTotalTxBytes)) {
        NS_LOG_WARN("networkInfo: Time difference is zero but byte count changed. This might indicate multiple calls within the same simulation tick or an issue. CurrentTotalRxBytes="
                    << currentTotalRxBytes << ", LastTotalRxBytes=" << lastTotalRxBytes
                    << ", CurrentTotalTxBytes=" << currentTotalTxBytes << ", LastTotalTxBytes=" << lastTotalTxBytes);
    } else {
        NS_LOG_DEBUG("networkInfo: Time difference is zero or negative (" << timeDiff << "s), not calculating throughput for this interval.");
    }

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<MobilityModel> mob = ueNodes.Get(i)->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        Vector vel = mob->GetVelocity();
        client_info[i].x_pos = pos.x;
        client_info[i].y_pos = pos.y;
        client_info[i].velocity = vel.x;

        auto [rsrp, sinr] = getRsrpSinr(i);

        NS_LOG_INFO("UE " << i << " | Pos: (" << pos.x << ", " << pos.y << ") | Vel: " << vel.x << " m/s | RSRP: " << rsrp << " | SINR: " << sinr);
    }

    lastTotalRxBytes = currentTotalRxBytes;
    lastTotalTxBytes = currentTotalTxBytes;
    lastTime = currentTime;
    NS_LOG_DEBUG("networkInfo: Updated lastTotalRxBytes=" << lastTotalRxBytes << ", lastTotalTxBytes=" << lastTotalTxBytes << ", lastTime=" << lastTime.GetSeconds() << "s.");
}

void MetricsCollector::logFlMetrics(int clientId, double accuracy) {
    fl_metrics_df.addRow({Simulator::Now().GetSeconds(), (double)roundNumber, (double)clientId, accuracy});
}

void MetricsCollector::exportDataFrames() {
    NS_LOG_INFO("Exporting DataFrames to 'results/' directory.");
    if (system("mkdir -p results") != 0) {
        NS_LOG_WARN("Could not create results directory.");
    }
    accuracy_df.toCsv("results/accuracy_fl_api.csv");
    participation_df.toCsv("results/clientParticipation_fl_api.csv");
    throughput_df.toCsv("results/throughput_fl_api.csv");
    rsrp_sinr_df.toCsv("results/rsrp_sinr_metrics.csv");
    fl_metrics_df.toCsv("results/fl_metrics.csv");
    NS_LOG_INFO("All DataFrames exported.");
}

void ReportUeSinrRsrp(uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId) {
    MetricsCollector::reportUeSinrRsrp(cellId, rnti, rsrp, sinr, componentCarrierId);
}

void ReportUeSinrRsrp(std::string context, uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId) {
    MetricsCollector::reportUeSinrRsrp(context, cellId, rnti, rsrp, sinr, componentCarrierId);
}