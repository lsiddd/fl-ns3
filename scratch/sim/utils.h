#ifndef SIM_UTILS_H
#define SIM_UTILS_H

// Include necessary libraries and modules
#include "client_types.h"
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

// Use NS3 namespace
using namespace ns3;

// Global variables declarations
extern std::map<Ipv4Address, double> endOfStreamTimes;
extern NodeContainer ueNodes;
extern NodeContainer remoteHostContainer;
extern std::map<uint16_t, std::map<uint16_t, double>> sinr_ue;
extern std::map<uint16_t, std::map<uint16_t, double>> rsrp_ue;
extern FlowMonitorHelper flowmon;

// Function declarations

// Utility function to extract RNTI and CellId from UE node
std::pair<uint16_t, uint16_t> get_ue_rnti_cellid(Ptr<ns3::NetDevice> ueNode);

// Function to map nodes to IPs
std::vector<NodesIps> node_to_ips();

// Report SINR and RSRP for a UE
void ReportUeSinrRsrp(uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId);

// Function to send a stream of data between nodes
void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size);

// Callback function to handle received packets
void RxCallback(const std::string path, Ptr<const Packet> packet, const Address& from);

// Run an external script and measure the time taken
int64_t runScriptAndMeasureTime(const std::string& scriptPath);

// Extract the model path from the input string
std::string extractModelPath(const std::string& input);

// Get the size of a file
std::streamsize getFileSize(const std::string& filename);

// Check if transmission has finished for given nodes and clients
bool finished_transmission(std::vector<NodesIps> nodes_ips,
                           std::vector<Clients_Models>& clients_info);

// Output network statistics based on the flow monitor
void network_info(Ptr<FlowMonitor> monitor);

// Clean up resources after a simulation round
void round_cleanup();

#endif // SIM_UTILS_H
