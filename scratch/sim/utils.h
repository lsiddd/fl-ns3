#pragma once

// Include necessary libraries and modules
#include "client_types.h"
#include "dataframe.h"

#include "ns3/applications-module.h"
#include "ns3/command-line.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h" // For Ptr, NodeContainer etc.
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/lte-ue-rrc.h" // Make sure LteUeRrc is included
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/log.h" // Include ns-3 log module


// Use NS3 namespace
using namespace ns3;

// Global variables declarations (ensure these match simulation.cc)
extern std::map<Ipv4Address, double> endOfStreamTimes;
extern NodeContainer ueNodes; // Used in nodeToIps, roundCleanup
extern NodeContainer remoteHostContainer; // Used in roundCleanup
extern std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
extern std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;
extern FlowMonitorHelper flowmon; // Used in networkInfo

extern DataFrame throughput_df; // Used in networkInfo
extern DataFrame rsrp_sinr_df; // New: Used in networkInfo to potentially log more RSRP/SINR if needed, though simulation.cc handles it.

// Function declarations

std::pair<uint16_t, uint16_t> getUeRntiCellid(Ptr<ns3::NetDevice> ueNode);

// Original RSRP/SINR callback (5 args) - may not be usable if trace source changed
void ReportUeSinrRsrp(uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);
// Overload for Config::Connect (with context string) - may not be usable
void ReportUeSinrRsrp(std::string context, uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);

// New callback for ReportCurrentCellRsrpSinr if it expects 3 arguments
void ReportUePhyMetricsFromTrace(unsigned long arg1, unsigned short arg2, unsigned short arg3);


std::vector<NodesIps> nodeToIps(); // Placed here as it's a utility for IP mapping

void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size);
void sinkRxCallback(Ptr<const Packet> packet, const Address& from); // Declaration for sinkRxCallback

// runScriptAndMeasureTime might not be directly used for API calls now, but keep it.
int64_t runScriptAndMeasureTime(const std::string& scriptPath);
std::string extractModelPath(const std::string& input); // Potentially unused
std::streamsize getFileSize(const std::string& filename); // Potentially unused

// Modified checkFinishedTransmission to take currently selected clients for the round
bool checkFinishedTransmission(const std::vector<NodesIps>& all_nodes_ips,
                               const std::vector<ClientModels>& selected_clients_for_round);

void networkInfo(Ptr<FlowMonitor> monitor);
void roundCleanup();
