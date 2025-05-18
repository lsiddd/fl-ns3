#define LOG(x) std::cout << x << std::endl 
#include "utils.h"
#include "MyApp.h"  // For MyApp in roundCleanup
#include "json.hpp" // If any JSON utilities are needed here

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
#include <iomanip> // For std::fixed and std::setprecision if needed for float logging

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

static const uint32_t writeSize = 2500; // Max packet payload size for MyApp
uint8_t data[writeSize];                // Filled with 'g'
uint8_t dataFin[writeSize];             // Filled with 'b'

// Initialize data arrays
struct DataInitializer
{
    DataInitializer()
    {
        // Using LOG directly as LOG in a global ctor might be tricky
        // depending on when logging is initialized. A simple LOG is safer here.
        // LOG("DataInitializer: Constructor invoked.");
        // LOG("DataInitializer: Initializing global data arrays 'data' and 'dataFin'.");
        // LOG("DataInitializer: Filling 'data' array (size " << writeSize << ") with 'g'.");
        std::fill_n(data, writeSize, 'g');
        // LOG("DataInitializer: Filling 'dataFin' array (size " << writeSize << ") with 'b'.");
        std::fill_n(dataFin, writeSize, 'b');
        // LOG("DataInitializer: Global data arrays 'data' and 'dataFin' initialized successfully.");
    }
};
DataInitializer globalDataInitializer; // This will trigger the constructor log

std::pair<uint16_t, uint16_t> getUeRntiCellid(Ptr<ns3::NetDevice> ueNetDevice)
{
    // // LOG(ueNetDevice.GetPointer()); // Log function entry with Ptr address
    // // LOG("Attempting to get RNTI and CellID for NetDevice: " << (ueNetDevice ? ueNetDevice->GetInstanceTypeId().GetName() : "null"));

    if (!ueNetDevice)
    {
        // // LOG("Input ueNetDevice is null. Returning {0, 0}.");
        return {0, 0};
    }

    auto lteDevice = ueNetDevice->GetObject<LteUeNetDevice>();
    if (!lteDevice)
    {
        // // LOG("NetDevice is not an LteUeNetDevice or GetObject failed. Device Type: " << ueNetDevice->GetInstanceTypeId().GetName() << ". Returning {0, 0}.");
        return {0, 0};
    }
    
    // // LOG("LteUeNetDevice found: " << lteDevice);

    if (!lteDevice->GetRrc())
    {
        // // LOG("LteUeNetDevice " << lteDevice << " has no RRC instance. Returning {0, 0}.");
        return {0, 0};
    }

    LteUeRrc::State rrcState = lteDevice->GetRrc()->GetState();
    // // LOG("LteUeNetDevice RRC State: " << rrcState);

    if (rrcState != LteUeRrc::CONNECTED_NORMALLY && rrcState != LteUeRrc::CONNECTED_HANDOVER)
    {
        // // LOG("UE RRC not in a connected state (Current state: " << rrcState << "). Returning {0, 0}.");
        return {0, 0};
    }

    auto rnti = lteDevice->GetRrc()->GetRnti();
    auto cellid = lteDevice->GetRrc()->GetCellId();
    // // LOG("Successfully retrieved RNTI: " << rnti << ", CellID: " << cellid << " for UE connected to eNB.");
    return std::make_pair(rnti, cellid);
}

void ReportUeSinrRsrp(uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId)
{
    // LOG(cellId << rnti << rsrp << sinr << componentCarrierId); // Log function entry with parameters
    // // LOG("ReportUeSinrRsrp (5 args) - CellID: " << cellId << ", RNTI: " << rnti
    //             << ", RSRP: " << std::fixed << std::setprecision(2) << rsrp << " dBm"
    //             << ", SINR: " << std::fixed << std::setprecision(2) << sinr << " dB"
    //             << ", CC ID: " << (unsigned int)componentCarrierId);

    // Assuming sinrUe and rsrpUe are global maps
    // std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
    // std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;
    sinrUe[cellId][rnti] = sinr;
    rsrpUe[cellId][rnti] = rsrp;
    // // LOG("Stored SINR=" << sinr << " and RSRP=" << rsrp << " for CellID=" << cellId << ", RNTI=" << rnti);
}

void ReportUeSinrRsrp(std::string context,
                      uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId)
{
    // LOG(context << cellId << rnti << rsrp << sinr << componentCarrierId); // Log function entry
    // // LOG("ReportUeSinrRsrp (context version) - Context: '" << context
    //             << "', CellID: " << cellId << ", RNTI: " << rnti
    //             << ", RSRP: " << std::fixed << std::setprecision(2) << rsrp << " dBm"
    //             << ", SINR: " << std::fixed << std::setprecision(2) << sinr << " dB"
    //             << ", CC ID: " << (unsigned int)componentCarrierId);
    
    // // LOG("Calling non-context version of ReportUeSinrRsrp.");
    ReportUeSinrRsrp(cellId, rnti, rsrp, sinr, componentCarrierId); // Call the non-context version
    // // LOG("Returned from non-context version of ReportUeSinrRsrp.");
}

void ReportUePhyMetricsFromTrace(unsigned long arg1, unsigned short arg2, unsigned short arg3)
{
    // LOG(arg1 << arg2 << arg3); // Log function entry
    // // LOG("ReportUePhyMetricsFromTrace (3 args) invoked. Arg1: " << arg1 << ", Arg2: " << (unsigned int)arg2 << ", Arg3: " << (unsigned int)arg3
    //             << ". The interpretation of these arguments is UNCERTAIN for RSRP/SINR. "
    //             << "Actual metrics depend on the specific trace source signature in your ns-3 LTE version. "
    //             << "This callback may NOT provide RSRP/SINR directly and current implementation does NOT populate sinrUe/rsrpUe maps. "
    //             << "Investigate fl-ns3/src/lte/model/lte-ue-phy.cc for 'ReportCurrentCellRsrpSinr' trace source details.");
    // At this point, we CANNOT populate sinrUe and rsrpUe maps with actual RSRP/SINR
    // unless these arguments somehow contain or allow deriving that information.
    // The logic in getRsrpSinr() in simulation.cc will be significantly affected
    // and likely will not provide correct RSRP/SINR values.
}


std::vector<NodesIps> nodeToIps()
{
    // LOG("Starting to map UE Node IDs to their IP addresses.");

    std::vector<NodesIps> nodes_ips_list;
    // LOG("Iterating over " << ueNodes.GetN() << " UE nodes.");

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        // LOG("Processing UE Node " << i << " (Node ID: " << (ueNode ? ueNode->GetId() : -1) << ")");

        if (!ueNode) {
            // LOG("UE Node at index " << i << " is null. Skipping.");
            continue;
        }

        Ptr<Ipv4> ipv4 = ueNode->GetObject<Ipv4>();
        if (ipv4)
        {
            // LOG("Node " << ueNode->GetId() << " has Ipv4 object. Number of interfaces: " << ipv4->GetNInterfaces());
            if (ipv4->GetNInterfaces() > 1) // Assuming interface 1 is the LTE interface
            {
                Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0); // Get first address of interface 1
                Ipv4Address ipAddr = iaddr.GetLocal();
                // LOG("Node ID: " << ueNode->GetId() << ", UE Index: " << i << ", IP Address: " << ipAddr);
                nodes_ips_list.push_back(NodesIps(ueNode->GetId(), i, ipAddr));
            }
            else
            {
                // LOG("Node " << ueNode->GetId() << " (UE Index " << i << ") has Ipv4 object but less than 2 interfaces (found "
                            // <<ipv4->GetNInterfaces() << "). Cannot get IP from interface 1.");
            }
        }
        else
        {
            // LOG("Node " << ueNode->GetId() << " (UE Index " << i << ") does not have an Ipv4 object. Cannot get IP.");
        }
    }
    // LOG("Finished mapping. Found " << nodes_ips_list.size() << " UE nodes with IP addresses on interface 1.");
    return nodes_ips_list;
}

void sinkRxCallback(Ptr<const Packet> packet, const Address &from)
{
    // // LOG(packet.GetPointer() << &from); // Log function entry with Ptr address and Address ref
    
    if (!packet) {
        // // LOG("sinkRxCallback: Received a null packet. Ignoring.");
        return;
    }
    uint32_t packetSize = packet->GetSize();
    // // LOG("sinkRxCallback: Received packet of size " << packetSize << " from " << InetSocketAddress::ConvertFrom(from));

    if (packetSize == 0)
    {
        // // LOG("sinkRxCallback: Received empty packet (size 0). Ignoring.");
        return;
    }

    std::vector<uint8_t> buffer(packetSize);
    packet->CopyData(buffer.data(), packetSize);
    // // LOG("sinkRxCallback: Copied " << packetSize << " bytes to local buffer.");

    InetSocketAddress address = InetSocketAddress::ConvertFrom(from);
    Ipv4Address senderIp = address.GetIpv4();
    // // LOG("sinkRxCallback: Sender IP: " << senderIp);

    bool fin_packet = false;
    // // LOG("sinkRxCallback: Scanning packet payload for 'b' character (FIN signal)...");
    for (uint32_t i = 0; i < packetSize; ++i)
    {
        if (buffer[i] == 'b')
        {
            fin_packet = true;
            // LOG("sinkRxCallback: 'b' character (FIN signal) found in packet from " << senderIp);
            break;
        }
    }

    if (fin_packet)
    {
        double receiveTime = Simulator::Now().GetSeconds();
        // LOG(std::fixed << std::setprecision(6) << receiveTime << "s: Stream ending signal ('b') received from " << senderIp);
        if (endOfStreamTimes.find(senderIp) == endOfStreamTimes.end())
        {
            endOfStreamTimes[senderIp] = receiveTime;
            // LOG("sinkRxCallback: Recorded end-of-stream time " << receiveTime << "s for sender " << senderIp);
        }
        else
        {
            // LOG("sinkRxCallback: Received duplicate FIN signal from " << senderIp 
                        // <<". Original time: " << endOfStreamTimes[senderIp] << "s, new time: " << receiveTime << "s. Not updating.");
        }
    } else {
        // LOG("sinkRxCallback: Packet from " << senderIp << " is a data packet (no 'b' found).");
    }
}

void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size)
{
    // LOG((sendingNode ? sendingNode->GetId() : -1) << (receivingNode ? receivingNode->GetId() : -1) << size);
    
    if (!sendingNode || !receivingNode) {
        // LOG("sendStream: One or both nodes are null. Sending Node: " << sendingNode << ", Receiving Node: " << receivingNode << ". Aborting stream setup.");
        return;
    }
    // LOG("sendStream: Initiating stream from Node " << sendingNode->GetId() << " to Node " << receivingNode->GetId() << " for " << size << " bytes.");

    static uint16_t port_counter = 5000; // Static port increment
    uint16_t current_port = port_counter++; // Use a unique port for this stream's sink
    // LOG("sendStream: Using port " << current_port << " for this stream.");


    if (size == 0)
    {
        // LOG("sendStream: Requested size is 0. Nothing to send from Node " << sendingNode->GetId() << " to Node " << receivingNode->GetId() << ". Skipping stream setup.");
        return;
    }

    uint32_t nPackets = (size + writeSize - 1) / writeSize;
    if (nPackets == 0 && size > 0) { 
        nPackets = 1;
        // LOG("sendStream: Size " << size << " is less than writeSize " << writeSize << ", adjusting nPackets to 1.");
    }
    // LOG("sendStream: Calculated nPackets = " << nPackets << " for total size " << size << " with writeSize " << writeSize);

    Ptr<Ipv4> ipv4Receiver = receivingNode->GetObject<Ipv4>();
    if (!ipv4Receiver)
    {
        // LOG("sendStream: No Ipv4 object found on receiving node " << receivingNode->GetId() << ". Cannot determine destination IP. Aborting.");
        // NS_FATAL_ERROR can be used here if this is unrecoverable
        return;
    }
    
    if (ipv4Receiver->GetNInterfaces() <= 1) {
         // LOG("sendStream: Receiving node " << receivingNode->GetId() << " has " << ipv4Receiver->GetNInterfaces() 
                      // <<" Ipv4 interfaces. Expected >1 for LTE context (if 1 is the UE interface). Aborting.");
        return;
    }
    // Assuming interface 1 is the relevant one for reaching the remote host via PGW.
    Ipv4InterfaceAddress iaddr = ipv4Receiver->GetAddress(1, 0); // Assuming interface 1, address 0
    Ipv4Address ipAddrReceiver = iaddr.GetLocal();
    // LOG("sendStream: Receiving Node " << receivingNode->GetId() << " IP Address (interface 1): " << ipAddrReceiver);

    // LOG(Simulator::Now().GetSeconds()
        // <<"s: Node " << sendingNode->GetId() << " starting stream to "
        // <<receivingNode->GetId() << " (" << ipAddrReceiver << ":" << current_port << "), " << size << " bytes, " 
        // <<nPackets << " packets, each up to " << writeSize << " payload bytes.");

    Address sinkAddress(InetSocketAddress(ipAddrReceiver, current_port));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), current_port));
    // LOG("sendStream: Installing TCP PacketSink on Node " << receivingNode->GetId() << " at " << InetSocketAddress(Ipv4Address::GetAny(), current_port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receivingNode);
    sinkApps.Start(Simulator::Now());
    // LOG("sendStream: PacketSink application started at " << Simulator::Now().GetSeconds() << "s.");
    // Consider stopping sinkApps explicitly later if needed.

    if (sinkApps.GetN() == 0) {
        // LOG("sendStream: Failed to install PacketSink on Node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApps.Get(0));
    if (!sink) {
        // LOG("sendStream: Failed to cast Application to PacketSink on Node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    sink->TraceConnectWithoutContext("Rx", MakeCallback(&sinkRxCallback));
    // LOG("sendStream: Connected Rx trace of PacketSink on Node " << receivingNode->GetId() << " to sinkRxCallback.");

    // LOG("sendStream: Creating TCP socket on sending Node " << sendingNode->GetId());
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(sendingNode, TcpSocketFactory::GetTypeId());
    if (!ns3TcpSocket)
    {
        // LOG("sendStream: Failed to create TCP socket on sending Node " << sendingNode->GetId() << ". Aborting.");
        // NS_FATAL_ERROR can be used here
        return;
    }
    // LOG("sendStream: TCP socket created successfully on Node " << sendingNode->GetId());

    // LOG("sendStream: Creating MyApp application object.");
    Ptr<MyApp> app = CreateObject<MyApp>();
    // LOG("sendStream: Setting up MyApp on Node " << sendingNode->GetId() << " to send to " << sinkAddress
                // <<", packetSizeForScheduling=" << writeSize << ", nPackets=" << nPackets 
                // <<", dataRate=10Mbps, actualSendSize=" << writeSize);
    app->Setup(ns3TcpSocket, sinkAddress, writeSize, nPackets, DataRate("10Mbps"), writeSize, data, dataFin);
    
    sendingNode->AddApplication(app);
    // LOG("sendStream: MyApp added to Node " << sendingNode->GetId());
    app->SetStartTime(Simulator::Now());
    // LOG("sendStream: MyApp started at " << Simulator::Now().GetSeconds() << "s on Node " << sendingNode->GetId());
    // Consider stopping app explicitly later.
}

std::streamsize getFileSize(const std::string &filename)
{
    // LOG(filename);
    // LOG("Attempting to get size of file: '" << filename << "'");

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        // LOG("Could not open file: '" << filename << "'. Returning size 0.");
        return 0;
    }
    std::streamsize size = file.tellg();
    // LOG("File '" << filename << "' size: " << size << " bytes.");
    file.close();
    return size;
}

std::string extractModelPath(const std::string &input)
{
    // LOG(input);
    // LOG("Extracting model path from input string: '" << input << "'");

    const std::string modelFlag = "--model ";
    const std::string extension = ".keras";
    const std::string modelSuffix = "_model";

    size_t modelPos = input.find(modelFlag);
    // LOG("Searching for '" << modelFlag << "'. Found at pos: " << modelPos);
    if (modelPos == std::string::npos) {
        // LOG("'" << modelFlag << "' not found. Returning empty string.");
        return "";
    }
    
    size_t start = modelPos + modelFlag.length();
    size_t end = input.find(" ", start);
    if (end == std::string::npos) {
        // LOG("No space found after model path, using end of string.");
        end = input.length();
    }
    // LOG("Model path candidate start: " << start << ", end: " << end);

    std::string modelPath = input.substr(start, end - start);
    // LOG("Initial extracted model path: '" << modelPath << "'");

    size_t extensionPos = modelPath.find(extension);
    // LOG("Searching for extension '" << extension << "'. Found at pos: " << extensionPos);
    if (extensionPos != std::string::npos) {
        modelPath = modelPath.substr(0, extensionPos);
        // LOG("Removed extension. Path now: '" << modelPath << "'");
    }

    size_t modelSuffixPos = modelPath.rfind(modelSuffix);
    // LOG("Searching for suffix '" << modelSuffix << "'. Found at pos: " << modelSuffixPos);
    if (modelSuffixPos != std::string::npos) {
        modelPath = modelPath.substr(0, modelSuffixPos);
        // LOG("Removed suffix. Path now: '" << modelPath << "'");
    }
    
    // LOG("Final extracted model path: '" << modelPath << "' from input '" << input << "'");
    return modelPath;
}

int64_t runScriptAndMeasureTime(const std::string &scriptPath)
{
    // LOG(scriptPath);
    // LOG("Preparing to run script: '" << scriptPath << "' and measure execution time.");

    auto startTime = std::chrono::high_resolution_clock::now();
    // LOG("Start time recorded.");

    std::string modelPath = extractModelPath(scriptPath);
    // LOG("Extracted model path for output redirection: '" << modelPath << "'");

    std::string cmdOutputFile;
    std::string command = "python3 " + scriptPath;
    if (!modelPath.empty())
    {
        cmdOutputFile = modelPath + ".txt";
        command += " > " + cmdOutputFile;
        // LOG("Output will be redirected to: '" << cmdOutputFile << "'");
    } else {
        // LOG("No model path extracted, script output will not be redirected to a specific file.");
    }

    // LOG("Executing command: '" << command << "'");
    int result = system(command.c_str());
    auto endTime = std::chrono::high_resolution_clock::now();
    // LOG("End time recorded. System call result: " << result);

    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    if (result != 0)
    {
        // LOG("Error: Python script '" << scriptPath << "' execution failed with exit code " << result 
                     // <<". Duration: " << duration << " ms.");
        return -1; // Indicate error
    }
    
    // LOG("Python script '" << scriptPath << "' executed successfully in " << duration << " ms.");
    return duration;
}

bool checkFinishedTransmission(const std::vector<NodesIps> &all_nodes_ips,
                               const std::vector<ClientModels> &selected_clients_for_round)
{
    // // LOG(); // Parameters are complex, logging them individually
    // LOG("Checking finished transmissions. Number of all_nodes_ips: " << all_nodes_ips.size()
                // <<", Number of selected_clients_for_round: " << selected_clients_for_round.size());

    if (selected_clients_for_round.empty())
    {
        // LOG("No clients selected for this round. Transmission considered finished.");
        return true;
    }

    size_t finished_count = 0;
    // LOG("Iterating through " << selected_clients_for_round.size() << " selected clients to check transmission status.");
    for (const auto &client_model : selected_clients_for_round)
    {
        if (!client_model.node) {
            // LOG("Client model has a null Ptr<Node>. Skipping this client.");
            continue;
        }
        uint32_t clientNodeId = client_model.node->GetId();
        // LOG("Checking client with Node ID: " << clientNodeId);

        Ipv4Address client_ip;
        bool ip_found = false;
        for (const auto &nip : all_nodes_ips)
        {
            if (nip.nodeId == clientNodeId)
            {
                client_ip = nip.ip;
                ip_found = true;
                // LOG("Found IP " << client_ip << " for Node ID " << clientNodeId);
                break;
            }
        }

        if (ip_found)
        {
            if (endOfStreamTimes.count(client_ip))
            {
                // LOG("Client " << clientNodeId << " (IP: " << client_ip << ") has finished transmission (EOS time found: " << endOfStreamTimes.at(client_ip) << "s).");
                finished_count++;
            }
            else
            {
                // LOG("Client " << clientNodeId << " (IP: " << client_ip << ") has NOT finished transmission yet (no EOS time found).");
            }
        }
        else
        {
            // LOG("Could not find IP address for client Node ID " << clientNodeId << " in all_nodes_ips. Cannot check its transmission status.");
        }
    }
    
    bool all_finished = (finished_count == selected_clients_for_round.size());
    // LOG("Finished transmission check: " << finished_count << " out of " << selected_clients_for_round.size() 
                // <<" selected clients have completed. All finished: " << (all_finished ? "YES" : "NO"));
    return all_finished;
}

void networkInfo(Ptr<FlowMonitor> monitor)
{
    // // LOG(monitor.GetPointer());

    if (!monitor) {
        // LOG("networkInfo: FlowMonitor Ptr is null. Cannot gather stats.");
        // Reschedule to prevent breaking the chain if this is an error state we want to recover from or log periodically.
        // However, if it's null once, it'll likely be null again.
        // Consider if rescheduling is wise here or if it should just stop.
        // For now, let's reschedule as per original logic but log an error.
        Simulator::Schedule(Seconds(1.0), &networkInfo, monitor);
        return;
    }

    static Time lastTime = Seconds(0);
    static double lastTotalRxBytes = 0;

    // LOG("networkInfo: Scheduled at " << Simulator::Now().GetSeconds() << "s. Last time: " << lastTime.GetSeconds() 
                // <<"s, lastTotalRxBytes: " << lastTotalRxBytes);

    // Schedule next call
    Simulator::Schedule(Seconds(1.0), &networkInfo, monitor);
    // LOG("networkInfo: Scheduled next call for 1.0s from now.");

    monitor->CheckForLostPackets();
    // LOG("networkInfo: Called CheckForLostPackets().");
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    // LOG("networkInfo: Retrieved " << stats.size() << " flow stats entries.");
    
    double currentTotalRxBytes = 0;
    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        // i->first is FlowId, i->second is FlowStats
        // LOG("Flow ID: " << i->first << ", RxBytes: " << i->second.rxBytes << ", TxBytes: " << i->second.txBytes);
        currentTotalRxBytes += i->second.rxBytes;
    }
    // LOG("networkInfo: Current total Rx Bytes sum: " << currentTotalRxBytes);

    Time currentTime = Simulator::Now();
    double timeDiff = (currentTime - lastTime).GetSeconds();
    // LOG("networkInfo: Current time: " << currentTime.GetSeconds() << "s. Time difference: " << timeDiff << "s.");

    if (timeDiff > 0)
    {
        // double instantRxThroughputMbps = (currentTotalRxBytes - lastTotalRxBytes) * 8.0 / timeDiff / 1e6;
        // LOG(currentTime.GetSeconds() << "s: Instant Rx Throughput: " << std::fixed << std::setprecision(4) << instantRxThroughputMbps << " Mbps");
        
        // Assuming throughput_df is a global object with an addRow method.
        // throughput_df.addRow({currentTime.GetSeconds(), 0.0 /*Tx placeholder*/, instantRxThroughputMbps});
        // LOG("networkInfo: Added row to throughput_df: Time=" << currentTime.GetSeconds() << ", RxThroughput=" << instantRxThroughputMbps);
    }
    else if (currentTime == lastTime && currentTotalRxBytes != lastTotalRxBytes)
    {
        // LOG("networkInfo: Time difference is zero but byte count changed. This might indicate multiple calls within the same simulation tick or an issue. currentTotalRxBytes="
                    // <<currentTotalRxBytes << ", lastTotalRxBytes=" << lastTotalRxBytes);
    }
    else
    {
         // LOG("networkInfo: Time difference is zero or negative (" << timeDiff << "s), not calculating throughput.");
    }


    lastTotalRxBytes = currentTotalRxBytes;
    lastTime = currentTime;
    // LOG("networkInfo: Updated lastTotalRxBytes=" << lastTotalRxBytes << ", lastTime=" << lastTime.GetSeconds() << "s.");
}

void roundCleanup()
{
    // LOG(Simulator::Now().GetSeconds() << "s: Starting round cleanup: Clearing endOfStreamTimes and stopping applications.");

    // LOG("Clearing endOfStreamTimes map. Current size: " << endOfStreamTimes.size());
    endOfStreamTimes.clear();
    // LOG("endOfStreamTimes map cleared.");

    // LOG("Iterating through " << ueNodes.GetN() << " UE nodes to stop MyApp applications.");
    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        if (!ueNode) {
            // LOG("UE Node at index " << i << " is null. Skipping app cleanup for this node.");
            continue;
        }
        // LOG("Processing UE Node " << ueNode->GetId() << " (index " << i << ").");
        // uint32_t numApps = ueNode->GetNApplications();
        // LOG("Node " << ueNode->GetId() << " has " << numApps << " applications.");
        
        // Iterate carefully as removing applications can change indices or count
        for (uint32_t j = 0; j < ueNode->GetNApplications(); /* no increment here */)
        {
            Ptr<Application> app = ueNode->GetApplication(j);
            if (!app) {
                 // LOG("Node " << ueNode->GetId() << ", App index " << j << " is null. Incrementing j and continuing.");
                 j++;
                 continue;
            }
            // LOG("Node " << ueNode->GetId() << ", checking App " << j << " (Type: " << app->GetInstanceTypeId().GetName() << ")");
            if (DynamicCast<MyApp>(app))
            {
                // LOG("Found MyApp on Node " << ueNode->GetId() << ". Setting StopTime to " << Simulator::Now().GetSeconds() << "s.");
                app->SetStopTime(Simulator::Now()); 
                // If MyApp removes itself or leads to re-indexing, this loop needs care.
                // Assuming SetStopTime just flags it and simulation handles removal, so j can be incremented.
                // If apps are removed immediately, iterate backwards or re-evaluate numApps.
                // For ns-3, SetStopTime usually doesn't remove it immediately from the node's list.
                j++; 
            }
            else
            {
                // LOG("App is not MyApp. Skipping.");
                j++;
            }
        }
    }
    // LOG("Finished processing MyApp applications on UE nodes.");

    if (remoteHostContainer.GetN() > 0)
    {
        Ptr<Node> serverNode = remoteHostContainer.Get(0); // Assuming one server
        if (!serverNode) {
             // LOG("Server node (remoteHostContainer.Get(0)) is null. Skipping PacketSink cleanup.");
        } else {
            // LOG("Processing server Node " << serverNode->GetId() << " to stop PacketSink applications.");
            // uint32_t numApps = serverNode->GetNApplications();
            // LOG("Server Node " << serverNode->GetId() << " has " << numApps << " applications.");
            
            for (uint32_t j = 0; j < serverNode->GetNApplications(); /* no increment */)
            {
                Ptr<Application> app = serverNode->GetApplication(j);
                 if (!app) {
                    // LOG("Server Node " << serverNode->GetId() << ", App index " << j << " is null. Incrementing j and continuing.");
                    j++;
                    continue;
                }
                // LOG("Server Node " << serverNode->GetId() << ", checking App " << j << " (Type: " << app->GetInstanceTypeId().GetName() << ")");
                if (DynamicCast<PacketSink>(app))
                {
                    // LOG("Found PacketSink on server Node " << serverNode->GetId() << ". Setting StopTime to " << Simulator::Now().GetSeconds() << "s.");
                    app->SetStopTime(Simulator::Now());
                    j++;
                }
                else
                {
                    // LOG("App is not PacketSink. Skipping.");
                    j++;
                }
            }
            // LOG("Finished processing PacketSink applications on server node.");
        }
    } else {
        // LOG("No remote hosts in remoteHostContainer. Skipping PacketSink cleanup.");
    }
    
    // LOG(Simulator::Now().GetSeconds() << "s: ns-3 round cleanup finished.");
}
