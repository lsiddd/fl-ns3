// Filename: scratch/sim/utils.cc
// Removed #define LOG(x) as we are using NS_LOG system
#include "utils.h"
#include "MyApp.h"  // For MyApp in roundCleanup
// #include "json.hpp" // If any JSON utilities are needed here, include if used

#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/log.h" // Make sure log module is included


#include <chrono>
#include <iomanip> // For std::fixed and std::setprecision
#include <numeric> // For std::accumulate

// Define the logging component for this file
NS_LOG_COMPONENT_DEFINE("Utils");

using namespace ns3;

// These are assumed to be declared globally elsewhere (e.g., in your main simulation file)
// extern NodeContainer ueNodes;
// extern NodeContainer remoteHostContainer;
// extern std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
// extern std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;
// extern std::map<Ipv4Address, double> endOfStreamTimes;
// extern struct ThroughputDataFrame throughput_df; // Assuming a struct/class for this

// Increased TCP Segment Size in simulation.cc default config, so let's align writeSize.
static const uint32_t writeSize = 1448; // Max packet payload size for MyApp (aligned with TCP Segment Size default)
uint8_t data[writeSize];                // Filled with 'g'
uint8_t dataFin[writeSize];             // Filled with 'b'

// Initialize data arrays
struct DataInitializer
{
    DataInitializer()
    {
        NS_LOG_INFO("DataInitializer: Initializing global data arrays 'data' and 'dataFin'.");
        std::fill_n(data, writeSize, 'g');
        std::fill_n(dataFin, writeSize, 'b');
        NS_LOG_INFO("DataInitializer: Global data arrays 'data' and 'dataFin' initialized successfully.");
    }
};
DataInitializer globalDataInitializer; // This will trigger the constructor log

std::pair<uint16_t, uint16_t> getUeRntiCellid(Ptr<ns3::NetDevice> ueNetDevice)
{
    NS_LOG_DEBUG("getUeRntiCellid: Attempting to get RNTI and CellID for NetDevice: " << (ueNetDevice ? ueNetDevice->GetInstanceTypeId().GetName() : "null"));

    if (!ueNetDevice)
    {
        NS_LOG_DEBUG("getUeRntiCellid: Input ueNetDevice is null. Returning {0, 0}.");
        return {0, 0};
    }

    auto lteDevice = ueNetDevice->GetObject<LteUeNetDevice>();
    if (!lteDevice)
    {
        NS_LOG_DEBUG("getUeRntiCellid: NetDevice is not an LteUeNetDevice or GetObject failed. Device Type: " << ueNetDevice->GetInstanceTypeId().GetName() << ". Returning {0, 0}.");
        return {0, 0};
    }
    
    NS_LOG_DEBUG("getUeRntiCellid: LteUeNetDevice found: " << lteDevice);

    if (!lteDevice->GetRrc())
    {
        NS_LOG_DEBUG("getUeRntiCellid: LteUeNetDevice " << lteDevice << " has no RRC instance. Returning {0, 0}.");
        return {0, 0};
    }

    LteUeRrc::State rrcState = lteDevice->GetRrc()->GetState();
    NS_LOG_DEBUG("getUeRntiCellid: LteUeRrc State: " << rrcState);

    if (rrcState != LteUeRrc::CONNECTED_NORMALLY && rrcState != LteUeRrc::CONNECTED_HANDOVER)
    {
        NS_LOG_DEBUG("getUeRntiCellid: UE RRC not in a connected state (Current state: " << rrcState << "). Returning {0, 0}.");
        return {0, 0};
    }

    auto rnti = lteDevice->GetRrc()->GetRnti();
    auto cellid = lteDevice->GetRrc()->GetCellId();
    NS_LOG_DEBUG("getUeRntiCellid: Successfully retrieved RNTI: " << rnti << ", CellID: " << cellid << " for UE connected to eNB.");
    return std::make_pair(rnti, cellid);
}

void ReportUeSinrRsrp(uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId)
{
    // NS_LOG_INFO("ReportUeSinrRsrp (5 args) - CellID: " << cellId << ", RNTI: " << rnti
    //             << ", RSRP: " << std::fixed << std::setprecision(2) << rsrp << " dBm"
    //             << ", SINR: " << std::fixed << std::setprecision(2) << sinr << " dB"
    //             << ", CC ID: " << (unsigned int)componentCarrierId);

    sinrUe[cellId][rnti] = sinr;
    rsrpUe[cellId][rnti] = rsrp;
    NS_LOG_DEBUG("ReportUeSinrRsrp: Stored SINR=" << sinr << " and RSRP=" << rsrp << " for CellID=" << cellId << ", RNTI=" << rnti);
}

void ReportUeSinrRsrp(std::string context,
                      uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId)
{
    NS_LOG_DEBUG("ReportUeSinrRsrp (context version) - Context: '" << context
                << "', CellID: " << cellId << ", RNTI: " << rnti
                << ", RSRP: " << std::fixed << std::setprecision(2) << rsrp << " dBm"
                << ", SINR: " << std::fixed << std::setprecision(2) << sinr << " dB"
                << ", CC ID: " << (unsigned int)componentCarrierId);
    
    NS_LOG_DEBUG("ReportUeSinrRsrp: Calling non-context version of ReportUeSinrRsrp.");
    ReportUeSinrRsrp(cellId, rnti, rsrp, sinr, componentCarrierId); // Call the non-context version
    NS_LOG_DEBUG("ReportUeSinrRsrp: Returned from non-context version of ReportUeSinrRsrp.");
}

void ReportUePhyMetricsFromTrace(unsigned long arg1, unsigned short arg2, unsigned short arg3)
{
    NS_LOG_WARN("ReportUePhyMetricsFromTrace (3 args) invoked. Arg1: " << arg1 << ", Arg2: " << (unsigned int)arg2 << ", Arg3: " << (unsigned int)arg3
                << ". The interpretation of these arguments is UNCERTAIN for RSRP/SINR. "
                << "Actual metrics depend on the specific trace source signature in your ns-3 LTE version. "
                << "This callback may NOT provide RSRP/SINR directly and current implementation does NOT populate sinrUe/rsrpUe maps. "
                << "Investigate fl-ns3/src/lte/model/lte-ue-phy.cc for 'ReportCurrentCellRsrpSinr' trace source details.");
}


std::vector<NodesIps> nodeToIps()
{
    NS_LOG_INFO("nodeToIps: Starting to map UE Node IDs to their IP addresses.");

    std::vector<NodesIps> nodes_ips_list;
    NS_LOG_DEBUG("nodeToIps: Iterating over " << ueNodes.GetN() << " UE nodes.");

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        NS_LOG_DEBUG("nodeToIps: Processing UE Node " << i << " (Node ID: " << (ueNode ? ueNode->GetId() : -1) << ")");

        if (!ueNode) {
            NS_LOG_WARN("nodeToIps: UE Node at index " << i << " is null. Skipping.");
            continue;
        }

        Ptr<Ipv4> ipv4 = ueNode->GetObject<Ipv4>();
        if (ipv4)
        {
            NS_LOG_DEBUG("nodeToIps: Node " << ueNode->GetId() << " has Ipv4 object. Number of interfaces: " << ipv4->GetNInterfaces());
            if (ipv4->GetNInterfaces() > 1) // Assuming interface 1 is the LTE interface
            {
                Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0); // Get first address of interface 1
                Ipv4Address ipAddr = iaddr.GetLocal();
                NS_LOG_DEBUG("nodeToIps: Node ID: " << ueNode->GetId() << ", UE Index: " << i << ", IP Address: " << ipAddr);
                nodes_ips_list.push_back(NodesIps(ueNode->GetId(), i, ipAddr));
            }
            else
            {
                NS_LOG_WARN("nodeToIps: Node " << ueNode->GetId() << " (UE Index " << i << ") has Ipv4 object but less than 2 interfaces (found "
                            <<ipv4->GetNInterfaces() << "). Cannot get IP from interface 1.");
            }
        }
        else
        {
            NS_LOG_WARN("nodeToIps: Node " << ueNode->GetId() << " (UE Index " << i << ") does not have an Ipv4 object. Cannot get IP.");
        }
    }
    NS_LOG_INFO("nodeToIps: Finished mapping. Found " << nodes_ips_list.size() << " UE nodes with IP addresses on interface 1.");
    return nodes_ips_list;
}

void sinkRxCallback(Ptr<const Packet> packet, const Address &from)
{
    NS_LOG_INFO("sinkRxCallback: Received packet of size " << packet->GetSize() << ". Ptr=" << packet << ", From=" << InetSocketAddress::ConvertFrom(from) << " at " << Simulator::Now().GetSeconds() << "s.");
    
    if (!packet) {
        NS_LOG_WARN("sinkRxCallback: Received a null packet. Ignoring.");
        return;
    }
    uint32_t packetSize = packet->GetSize();
    NS_LOG_DEBUG("sinkRxCallback: Received packet of size " << packetSize << " from " << InetSocketAddress::ConvertFrom(from));
    
    if (packetSize == 0)
    {
        NS_LOG_WARN("sinkRxCallback: Received empty packet (size 0). Ignoring.");
        return;
    }

    // Only copy a small header portion to check for 'b'
    uint32_t bytes_to_check = std::min((uint32_t)100, packetSize); // Check first 100 bytes
    std::vector<uint8_t> buffer(bytes_to_check);
    packet->CopyData(buffer.data(), bytes_to_check); // Only copy up to bytes_to_check
    NS_LOG_DEBUG("sinkRxCallback: Copied first " << bytes_to_check << " bytes to local buffer.");

    InetSocketAddress address = InetSocketAddress::ConvertFrom(from);
    Ipv4Address senderIp = address.GetIpv4();
    NS_LOG_DEBUG("sinkRxCallback: Sender IP: " << senderIp);

    bool fin_packet = false;
    NS_LOG_DEBUG("sinkRxCallback: Scanning packet payload for 'b' character (FIN signal)...");
    for (uint32_t i = 0; i < bytes_to_check; ++i) // Only scan copied part
    {
        if (buffer[i] == 'b')
        {
            fin_packet = true;
            NS_LOG_INFO("sinkRxCallback: 'b' character (FIN signal) found in packet from " << senderIp);
            break;
        }
    }

    if (fin_packet)
    {
        double receiveTime = Simulator::Now().GetSeconds();
        NS_LOG_INFO(std::fixed << std::setprecision(6) << receiveTime << "s: Stream ending signal ('b') received from " << senderIp);
        if (endOfStreamTimes.find(senderIp) == endOfStreamTimes.end())
        {
            endOfStreamTimes[senderIp] = receiveTime; // Record time only for the first FIN packet
            NS_LOG_INFO("sinkRxCallback: Recorded end-of-stream time " << receiveTime << "s for sender " << senderIp);
        }
        else
        {
            NS_LOG_DEBUG("sinkRxCallback: Received duplicate FIN signal from " << senderIp 
                        <<" . Original time: " << endOfStreamTimes[senderIp] << "s, new time: " << receiveTime << "s. Not updating.");
        }
    } else {
        NS_LOG_DEBUG("sinkRxCallback: Packet from " << senderIp << " is a data packet (no 'b' found in first " << bytes_to_check << " bytes).");
    }
}

void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size)
{
    NS_LOG_INFO("sendStream: Called with sendingNode=" << (sendingNode ? sendingNode->GetId() : 0)
                                                << ", receivingNode=" << (receivingNode ? receivingNode->GetId() : 0)
                                                << ", size=" << size << " bytes.");
    
    if (!sendingNode || !receivingNode) {
        NS_LOG_ERROR("sendStream: One or both nodes are null. Sending Node: " << sendingNode << ", Receiving Node: " << receivingNode << ". Aborting stream setup.");
        return;
    }
    NS_LOG_INFO("sendStream: Initiating stream from Node " << sendingNode->GetId() << " to Node " << receivingNode->GetId() << " for " << size << " bytes.");

    static uint16_t port_counter = 5000; // Static port increment
    uint16_t current_port = port_counter++; // Use a unique port for this stream's sink
    NS_LOG_DEBUG("sendStream: Using port " << current_port << " for this stream.");


    if (size == 0)
    {
        NS_LOG_INFO("sendStream: Requested size is 0. Nothing to send from Node " << sendingNode->GetId() << " to Node " << receivingNode->GetId() << ". Skipping stream setup.");
        return;
    }

    uint32_t nPackets = (size + writeSize - 1) / writeSize;
    if (nPackets == 0 && size > 0) { 
        nPackets = 1; // For very small sizes less than writeSize, still send 1 packet.
        NS_LOG_DEBUG("sendStream: Size " << size << " is less than writeSize " << writeSize << ", adjusting nPackets to 1.");
    } else if (nPackets > 0 && size % writeSize == 0) {
        // If size is an exact multiple of writeSize, the last packet is data. Need one more for FIN.
        nPackets += 1;
        NS_LOG_DEBUG("sendStream: Size " << size << " is a multiple of writeSize " << writeSize << ", adjusting nPackets to " << nPackets << " for FIN signal.");
    } else {
         NS_LOG_DEBUG("sendStream: nPackets calculated: " << nPackets);
    }


    Ptr<Ipv4> ipv4Receiver = receivingNode->GetObject<Ipv4>();
    if (!ipv4Receiver)
    {
        NS_LOG_ERROR("sendStream: No Ipv4 object found on receiving node " << receivingNode->GetId() << ". Cannot determine destination IP. Aborting.");
        return;
    }
    
    // RemoteHost/FL Server should be on interface 0 (default loopback/first P2P interface)
    // or interface 1 if it has loopback and P2P
    // Let's assume the correct interface for remoteHost (FL server) is the one connected to PGW
    Ipv4Address ipAddrReceiver;
    if (receivingNode->GetId() == remoteHostContainer.Get(0)->GetId()) { // If it's the FL server
        ipAddrReceiver = ipv4Receiver->GetAddress(1, 0).GetLocal(); // Assuming P2P is interface 1
    } else { // For other nodes (e.g., UEs, though UEs send, not receive here)
        ipAddrReceiver = ipv4Receiver->GetAddress(1, 0).GetLocal(); // Assuming LTE is interface 1
    }

    if (ipAddrReceiver == Ipv4Address()) {
        NS_LOG_ERROR("sendStream: Failed to get valid IP address for receiving node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    NS_LOG_DEBUG("sendStream: Receiving Node " << receivingNode->GetId() << " IP Address: " << ipAddrReceiver);

    NS_LOG_INFO(Simulator::Now().GetSeconds()
        <<"s: Node " << sendingNode->GetId() << " starting stream to "
        <<receivingNode->GetId() << " (" << ipAddrReceiver << ":" << current_port << "), " << size << " bytes, " 
        <<nPackets << " packets, each up to " << writeSize << " payload bytes.");

    Address sinkAddress(InetSocketAddress(ipAddrReceiver, current_port));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), current_port));
    NS_LOG_DEBUG("sendStream: Installing TCP PacketSink on Node " << receivingNode->GetId() << " at " << InetSocketAddress(Ipv4Address::GetAny(), current_port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receivingNode);
    sinkApps.Start(Simulator::Now());
    NS_LOG_INFO("sendStream: PacketSink application started at " << Simulator::Now().GetSeconds() << "s on Node " << receivingNode->GetId() << ".");
    
    if (sinkApps.GetN() == 0) {
        NS_LOG_ERROR("sendStream: Failed to install PacketSink on Node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApps.Get(0));
    if (!sink) {
        NS_LOG_ERROR("sendStream: Failed to cast Application to PacketSink on Node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    sink->TraceConnectWithoutContext("Rx", MakeCallback(&sinkRxCallback));
    NS_LOG_DEBUG("sendStream: Connected Rx trace of PacketSink on Node " << receivingNode->GetId() << " to sinkRxCallback.");

    NS_LOG_DEBUG("sendStream: Creating TCP socket on sending Node " << sendingNode->GetId());
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(sendingNode, TcpSocketFactory::GetTypeId());
    if (!ns3TcpSocket)
    {
        NS_LOG_ERROR("sendStream: Failed to create TCP socket on sending Node " << sendingNode->GetId() << ". Aborting.");
        return;
    }
    NS_LOG_DEBUG("sendStream: TCP socket created successfully on Node " << sendingNode->GetId());

    NS_LOG_DEBUG("sendStream: Creating MyApp application object.");
    Ptr<MyApp> app = CreateObject<MyApp>();
    NS_LOG_INFO("sendStream: Setting up MyApp on Node " << sendingNode->GetId() << " to send to " << sinkAddress
                <<", packetSizeForScheduling=" << writeSize << ", nPackets=" << nPackets 
                <<", dataRate=10Mbps, actualSendSize=" << writeSize << " bytes.");
    app->Setup(ns3TcpSocket, sinkAddress, writeSize, nPackets, DataRate("10Mbps"), writeSize, data, dataFin);
    
    sendingNode->AddApplication(app);
    NS_LOG_DEBUG("sendStream: MyApp added to Node " << sendingNode->GetId());
    app->SetStartTime(Simulator::Now());
    NS_LOG_INFO("sendStream: MyApp started at " << Simulator::Now().GetSeconds() << "s on Node " << sendingNode->GetId());
}

std::streamsize getFileSize(const std::string &filename)
{
    NS_LOG_DEBUG("getFileSize: Attempting to get size of file: '" << filename << "'");

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        NS_LOG_ERROR("getFileSize: Could not open file: '" << filename << "'. Returning size 0.");
        return 0;
    }
    std::streamsize size = file.tellg();
    NS_LOG_DEBUG("getFileSize: File '" << filename << "' size: " << size << " bytes.");
    file.close();
    return size;
}

std::string extractModelPath(const std::string &input)
{
    NS_LOG_DEBUG("extractModelPath: Extracting model path from input string: '" << input << "'");

    const std::string modelFlag = "--model ";
    const std::string extension = ".keras";
    const std::string modelSuffix = "_model";

    size_t modelPos = input.find(modelFlag);
    NS_LOG_DEBUG("extractModelPath: Searching for '" << modelFlag << "'. Found at pos: " << modelPos);
    if (modelPos == std::string::npos) {
        NS_LOG_DEBUG("extractModelPath: '" << modelFlag << "' not found. Returning empty string.");
        return "";
    }
    
    size_t start = modelPos + modelFlag.length();
    size_t end = input.find(" ", start);
    if (end == std::string::npos) {
        NS_LOG_DEBUG("extractModelPath: No space found after model path, using end of string.");
        end = input.length();
    }
    NS_LOG_DEBUG("extractModelPath: Model path candidate start: " << start << ", end: " << end);

    std::string modelPath = input.substr(start, end - start);
    NS_LOG_DEBUG("extractModelPath: Initial extracted model path: '" << modelPath << "'");

    size_t extensionPos = modelPath.find(extension);
    NS_LOG_DEBUG("extractModelPath: Searching for extension '" << extension << "'. Found at pos: " << extensionPos);
    if (extensionPos != std::string::npos) {
        modelPath = modelPath.substr(0, extensionPos);
        NS_LOG_DEBUG("extractModelPath: Removed extension. Path now: '" << modelPath << "'");
    }

    size_t modelSuffixPos = modelPath.rfind(modelSuffix);
    NS_LOG_DEBUG("extractModelPath: Searching for suffix '" << modelSuffix << "'. Found at pos: " << modelSuffixPos);
    if (modelSuffixPos != std::string::npos) {
        modelPath = modelPath.substr(0, modelSuffixPos);
        NS_LOG_DEBUG("extractModelPath: Removed suffix. Path now: '" << modelPath << "'");
    }
    
    NS_LOG_DEBUG("extractModelPath: Final extracted model path: '" << modelPath << "' from input '" << input << "'");
    return modelPath;
}

int64_t runScriptAndMeasureTime(const std::string &scriptPath)
{
    NS_LOG_INFO("runScriptAndMeasureTime: Preparing to run script: '" << scriptPath << "' and measure execution time.");

    auto startTime = std::chrono::high_resolution_clock::now();
    NS_LOG_DEBUG("runScriptAndMeasureTime: Start time recorded.");

    std::string modelPath = extractModelPath(scriptPath);
    NS_LOG_DEBUG("runScriptAndMeasureTime: Extracted model path for output redirection: '" << modelPath << "'");

    std::string cmdOutputFile;
    std::string command = "python3 " + scriptPath;
    if (!modelPath.empty())
    {
        cmdOutputFile = modelPath + ".txt";
        command += " > " + cmdOutputFile;
        NS_LOG_DEBUG("runScriptAndMeasureTime: Output will be redirected to: '" << cmdOutputFile << "'");
    } else {
        NS_LOG_DEBUG("runScriptAndMeasureTime: No model path extracted, script output will not be redirected to a specific file.");
    }

    NS_LOG_INFO("runScriptAndMeasureTime: Executing command: '" << command << "'");
    int result = system(command.c_str());
    auto endTime = std::chrono::high_resolution_clock::now();
    NS_LOG_DEBUG("runScriptAndMeasureTime: End time recorded. System call result: " << result);

    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    if (result != 0)
    {
        NS_LOG_ERROR("runScriptAndMeasureTime: Python script '" << scriptPath << "' execution failed with exit code " << result 
                     <<". Duration: " << duration << " ms.");
        return -1; // Indicate error
    }
    
    NS_LOG_INFO("runScriptAndMeasureTime: Python script '" << scriptPath << "' executed successfully in " << duration << " ms.");
    return duration;
}

bool checkFinishedTransmission(const std::vector<NodesIps> &all_nodes_ips,
                               const std::vector<ClientModels> &selected_clients_for_round)
{
    NS_LOG_DEBUG("checkFinishedTransmission: Checking finished transmissions. Number of all_nodes_ips: " << all_nodes_ips.size()
                <<", Number of selected_clients_for_round: " << selected_clients_for_round.size());

    if (selected_clients_for_round.empty())
    {
        NS_LOG_INFO("checkFinishedTransmission: No clients selected for this round. Transmission considered finished.");
        return true;
    }

    size_t finished_count = 0;
    NS_LOG_DEBUG("checkFinishedTransmission: Iterating through " << selected_clients_for_round.size() << " selected clients to check transmission status.");
    for (const auto &client_model : selected_clients_for_round)
    {
        if (!client_model.node) {
            NS_LOG_WARN("checkFinishedTransmission: Client model has a null Ptr<Node>. Skipping this client.");
            continue;
        }
        uint32_t clientNodeId = client_model.node->GetId();
        NS_LOG_DEBUG("checkFinishedTransmission: Checking client with Node ID: " << clientNodeId);

        Ipv4Address client_ip;
        bool ip_found = false;
        for (const auto &nip : all_nodes_ips)
        {
            if (nip.nodeId == clientNodeId)
            {
                client_ip = nip.ip;
                ip_found = true;
                NS_LOG_DEBUG("checkFinishedTransmission: Found IP " << client_ip << " for Node ID " << clientNodeId);
                break;
            }
        }

        if (ip_found)
        {
            if (endOfStreamTimes.count(client_ip))
            {
                NS_LOG_DEBUG("checkFinishedTransmission: Client " << clientNodeId << " (IP: " << client_ip << ") has finished transmission (EOS time found: " << endOfStreamTimes.at(client_ip) << "s).");
                finished_count++;
            }
            else
            {
                NS_LOG_DEBUG("checkFinishedTransmission: Client " << clientNodeId << " (IP: " << client_ip << ") has NOT finished transmission yet (no EOS time found).");
            }
        }
        else
        {
            NS_LOG_WARN("checkFinishedTransmission: Could not find IP address for client Node ID " << clientNodeId << " in all_nodes_ips. Cannot check its transmission status.");
        }
    }
    
    bool all_finished = (finished_count == selected_clients_for_round.size());
    NS_LOG_INFO("checkFinishedTransmission: Finished transmission check: " << finished_count << " out of " << selected_clients_for_round.size() 
                <<" selected clients have completed. All finished: " << (all_finished ? "YES" : "NO"));
    return all_finished;
}

void networkInfo(Ptr<FlowMonitor> monitor)
{
    static Time lastTime = Seconds(0);
    static uint64_t lastTotalRxBytes = 0;
    static uint64_t lastTotalTxBytes = 0; // New: track total TX bytes

    NS_LOG_DEBUG("networkInfo: Scheduled at " << Simulator::Now().GetSeconds() << "s. Last time: " << lastTime.GetSeconds() 
                <<"s, lastTotalRxBytes: " << lastTotalRxBytes << ", lastTotalTxBytes: " << lastTotalTxBytes);

    // Schedule next call
    Simulator::Schedule(Seconds(1.0), &networkInfo, monitor);
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
    uint64_t currentTotalTxBytes = 0; // New: current total TX bytes

    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        // i->first is FlowId, i->second is FlowStats
        NS_LOG_DEBUG("Flow ID: " << i->first << ", RxBytes: " << i->second.rxBytes << ", TxBytes: " << i->second.txBytes
                     << ", Packets Rx: " << i->second.rxPackets << ", Packets Tx: " << i->second.txPackets
                     << ", Lost: " << i->second.lostPackets << ", Jitter: " << i->second.jitterSum.GetSeconds());
        currentTotalRxBytes += i->second.rxBytes;
        currentTotalTxBytes += i->second.txBytes; // New: accumulate TX bytes
    }
    NS_LOG_DEBUG("networkInfo: Current total Rx Bytes sum: " << currentTotalRxBytes << ", Current total Tx Bytes sum: " << currentTotalTxBytes);

    Time currentTime = Simulator::Now();
    double timeDiff = (currentTime - lastTime).GetSeconds();
    NS_LOG_DEBUG("networkInfo: Current time: " << currentTime.GetSeconds() << "s. Time difference: " << timeDiff << "s.");

    if (timeDiff > 0)
    {
        double instantRxThroughputMbps = static_cast<double>(currentTotalRxBytes - lastTotalRxBytes) * 8.0 / timeDiff / 1e6;
        double instantTxThroughputMbps = static_cast<double>(currentTotalTxBytes - lastTotalTxBytes) * 8.0 / timeDiff / 1e6; // New: calculate TX throughput
        
        NS_LOG_INFO(currentTime.GetSeconds() << "s: Instant Rx Throughput: " << std::fixed << std::setprecision(4) << instantRxThroughputMbps << " Mbps, "
                                             << "Instant Tx Throughput: " << std::fixed << std::setprecision(4) << instantTxThroughputMbps << " Mbps.");
        
        throughput_df.addRow({currentTime.GetSeconds(), instantTxThroughputMbps, instantRxThroughputMbps, (uint32_t)currentTotalTxBytes, (uint32_t)currentTotalRxBytes});
        NS_LOG_DEBUG("networkInfo: Added row to throughput_df: Time=" << currentTime.GetSeconds() 
                     << ", TxThroughput=" << instantTxThroughputMbps << ", RxThroughput=" << instantRxThroughputMbps
                     << ", TotalTxBytes=" << currentTotalTxBytes << ", TotalRxBytes=" << currentTotalRxBytes);
    }
    else if (currentTime == lastTime && (currentTotalRxBytes != lastTotalRxBytes || currentTotalTxBytes != lastTotalTxBytes))
    {
        NS_LOG_WARN("networkInfo: Time difference is zero but byte count changed. This might indicate multiple calls within the same simulation tick or an issue. CurrentTotalRxBytes="
                    <<currentTotalRxBytes << ", LastTotalRxBytes=" << lastTotalRxBytes 
                    << ", CurrentTotalTxBytes=" << currentTotalTxBytes << ", LastTotalTxBytes=" << lastTotalTxBytes);
    }
    else
    {
         NS_LOG_DEBUG("networkInfo: Time difference is zero or negative (" << timeDiff << "s), not calculating throughput for this interval.");
    }

    lastTotalRxBytes = currentTotalRxBytes;
    lastTotalTxBytes = currentTotalTxBytes; // New: update last TX bytes
    lastTime = currentTime;
    NS_LOG_DEBUG("networkInfo: Updated lastTotalRxBytes=" << lastTotalRxBytes << ", lastTotalTxBytes=" << lastTotalTxBytes << ", lastTime=" << lastTime.GetSeconds() << "s.");
}

void roundCleanup()
{
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: Starting round cleanup: Clearing endOfStreamTimes and stopping applications.");

    NS_LOG_INFO("roundCleanup: Clearing endOfStreamTimes map. Current size: " << endOfStreamTimes.size());
    endOfStreamTimes.clear();
    NS_LOG_INFO("roundCleanup: endOfStreamTimes map cleared.");

    NS_LOG_INFO("roundCleanup: Iterating through " << ueNodes.GetN() << " UE nodes to stop MyApp applications.");
    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        if (!ueNode) {
            NS_LOG_WARN("roundCleanup: UE Node at index " << i << " is null. Skipping app cleanup for this node.");
            continue;
        }
        NS_LOG_DEBUG("roundCleanup: Processing UE Node " << ueNode->GetId() << " (index " << i << "). Has " << ueNode->GetNApplications() << " applications.");
        
        // Iterate backwards to safely remove applications if needed, or if SetStopTime effectively removes them
        // For ns-3, SetStopTime usually just flags for stopping, not immediate removal from node's app list.
        for (int j = ueNode->GetNApplications() - 1; j >= 0; --j)
        {
            Ptr<Application> app = ueNode->GetApplication(j);
            if (!app) {
                 NS_LOG_WARN("roundCleanup: Node " << ueNode->GetId() << ", App index " << j << " is null. Skipping.");
                 continue;
            }
            NS_LOG_DEBUG("roundCleanup: Node " << ueNode->GetId() << ", checking App " << j << " (Type: " << app->GetInstanceTypeId().GetName() << ")");
            if (DynamicCast<MyApp>(app))
            {
                NS_LOG_INFO("roundCleanup: Found MyApp on Node " << ueNode->GetId() << ". Setting StopTime to " << Simulator::Now().GetSeconds() << "s.");
                app->SetStopTime(Simulator::Now()); 
            }
        }
    }
    NS_LOG_INFO("roundCleanup: Finished processing MyApp applications on UE nodes.");

    if (remoteHostContainer.GetN() > 0)
    {
        Ptr<Node> serverNode = remoteHostContainer.Get(0); // Assuming one server
        if (!serverNode) {
             NS_LOG_WARN("roundCleanup: Server node (remoteHostContainer.Get(0)) is null. Skipping PacketSink cleanup.");
        } else {
            NS_LOG_INFO("roundCleanup: Processing server Node " << serverNode->GetId() << " to stop PacketSink applications. Has " << serverNode->GetNApplications() << " applications.");
            
            for (int j = serverNode->GetNApplications() - 1; j >= 0; --j)
            {
                Ptr<Application> app = serverNode->GetApplication(j);
                 if (!app) {
                    NS_LOG_WARN("roundCleanup: Server Node " << serverNode->GetId() << ", App index " << j << " is null. Skipping.");
                    continue;
                }
                NS_LOG_DEBUG("roundCleanup: Server Node " << serverNode->GetId() << ", checking App " << j << " (Type: " << app->GetInstanceTypeId().GetName() << ")");
                if (DynamicCast<PacketSink>(app))
                {
                    NS_LOG_INFO("roundCleanup: Found PacketSink on server Node " << serverNode->GetId() << ". Setting StopTime to " << Simulator::Now().GetSeconds() << "s.");
                    app->SetStopTime(Simulator::Now());
                }
            }
            NS_LOG_INFO("roundCleanup: Finished processing PacketSink applications on server node.");
        }
    } else {
        NS_LOG_INFO("roundCleanup: No remote hosts in remoteHostContainer. Skipping PacketSink cleanup.");
    }
    
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: ns-3 round cleanup finished.");
}