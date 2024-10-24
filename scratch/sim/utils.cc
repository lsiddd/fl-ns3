#define LOG(x) std::cout << x << std::endl

#include "utils.h"

#include "MyApp.h"
#include "json.hpp"

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

#include <chrono>

using namespace ns3;

static const uint32_t writeSize = 2500;
uint8_t data[writeSize] = {'g'};
uint8_t data_fin[writeSize] = {'b'};

std::pair<uint16_t, uint16_t> get_ue_rnti_cellid(Ptr<ns3::NetDevice> ueNetDevice)
{
    auto rnti = ueNetDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetRnti();
    auto cellid = ueNetDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetCellId();
    return std::make_pair(rnti, cellid);
}

void ReportUeSinrRsrp(uint16_t cellId,
                      uint16_t rnti,
                      double rsrp,
                      double sinr,
                      uint8_t componentCarrierId)
{
    //   std::cout << "CellId: " << cellId << ", RNTI: " << rnti
    //             << ", RSRP: " << rsrp << " dBm, SINR: " << sinr << " dB "  << " cc Id: " <<
    //             componentCarrierId << std::endl;
    sinr_ue[cellId][rnti] = sinr;
    rsrp_ue[cellId][rnti] = rsrp;
}



std::vector<NodesIps>node_to_ips()
{
    std::vector<NodesIps> nodes_ips;

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ipv4Address receiving_address;
        Ptr<Ipv4> ipv4 = ueNodes.Get(i)->GetObject<Ipv4>();
        Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0);
        Ipv4Address ipAddr = iaddr.GetLocal();
        nodes_ips.push_back(NodesIps(ueNodes.Get(i)->GetId(), i, ipAddr));
    }

    return nodes_ips;
}

void RxCallback(const std::string path, Ptr<const Packet> packet, const Address& from)
{
    uint32_t packetSize = packet->GetSize();
    std::vector<uint8_t> buffer(packetSize);
    packet->CopyData(buffer.data(), packetSize);
    std::string dataAsString(reinterpret_cast<char*>(buffer.data()), packetSize);
    InetSocketAddress address = InetSocketAddress::ConvertFrom(from);
    Ipv4Address senderIp = address.GetIpv4();

    if (dataAsString.find('b') != std::string::npos) {
        double receiveTime = Simulator::Now().GetSeconds();
        std::cout << "Stream ending signal ('b') received from " << senderIp << " at time "
                  << receiveTime << " seconds." << std::endl;
        endOfStreamTimes[senderIp] = receiveTime; // Assuming endOfStreamTimes is declared somewhere
    }
}

void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size)
{
    static uint16_t port = 5000; // Initialized once and incremented for each call
    port++;
    constexpr int packetSize = 1040;  // Defined as a constant for better readability
    int nPackets = size / packetSize; // Calculate the number of packets
    // Get receiving node's IPv4 address
    Ptr<Ipv4> ipv4 = receivingNode->GetObject<Ipv4>();

    if (!ipv4) {
        NS_FATAL_ERROR("No Ipv4 object found on receiving node");
    }

    Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0); // Assume the second interface
    Ipv4Address ipAddr = iaddr.GetLocal();
    // Logging the stream initiation
    LOG(Simulator::Now().GetSeconds()
        << "s: Starting stream from node " << sendingNode->GetId() << " to node "
        << receivingNode->GetId() << ", " << size << " bytes.");
    // Create and configure the sink (receiver)
    Address sinkAddress(InetSocketAddress(ipAddr, port));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receivingNode);
    sinkApps.Start(Seconds(0.1));
    // Create the sending socket
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(sendingNode, TcpSocketFactory::GetTypeId());

    if (!ns3TcpSocket) {
        NS_FATAL_ERROR("Failed to create TCP socket on sending node");
    }

    // Create and configure the application to send packets
    Ptr<MyApp> app = CreateObject<MyApp>();
    app->Setup(ns3TcpSocket,
               sinkAddress,
               packetSize,
               nPackets,
               DataRate("5Mb/s"),
               writeSize,
               data,
               data_fin);
    sendingNode->AddApplication(app);
    // Schedule the start and stop times for the application
    app->SetStartTime(Seconds(0.5));
    // app->SetStopTime(Seconds(simStopTime));  // Assuming simStopTime is declared globally
    // Connect the callback for when packets are received at the sink
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback(&RxCallback));
}

// Utility Functions ==================================
// Function to get the size of a file
std::streamsize getFileSize(const std::string& filename)
{
    std::ifstream file(filename,
                       std::ios::binary | std::ios::ate); // Open file in binary mode at the end

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        return -1; // Return -1 to indicate an error
    }

    return file.tellg(); // Get the file size by reading the position of the end of the file
}

// Function to extract model path without suffix and extension
std::string extractModelPath(const std::string& input)
{
    const std::string modelFlag = "--model ";
    const std::string extension = ".keras";
    const std::string modelSuffix = "_model";
    size_t modelPos = input.find(modelFlag);

    if (modelPos == std::string::npos) {
        return ""; // Return empty string if "--model" flag is not found
    }

    // Start after the "--model " flag
    size_t start = modelPos + modelFlag.length();
    // Find the position of the next space after the model path
    size_t end = input.find(" ", start);

    if (end == std::string::npos) {
        end = input.length(); // If no space is found, assume the model path goes to the end
    }

    std::string modelPath = input.substr(start, end - start); // Extract model path
    // Remove the ".keras" extension if it exists
    size_t extensionPos = modelPath.find(extension);

    if (extensionPos != std::string::npos) {
        modelPath = modelPath.substr(0, extensionPos);
    }

    // Remove the "_model" suffix if it exists
    size_t modelSuffixPos = modelPath.rfind(modelSuffix);

    if (modelSuffixPos != std::string::npos) {
        modelPath = modelPath.substr(0, modelSuffixPos);
    }

    return modelPath;
}

// Function to run a Python script and measure its execution time
int64_t runScriptAndMeasureTime(const std::string& scriptPath)
{
    auto startTime = std::chrono::high_resolution_clock::now(); // Record start time
    std::string modelPath = extractModelPath(scriptPath);
    std::string cmdOutputFile = modelPath + ".txt";
    std::string command =
        "python3 " + scriptPath + " > " + cmdOutputFile + " 2>&1"; // Redirect output to a file
    int result = system(command.c_str());                     // Run the Python script
    auto endTime = std::chrono::high_resolution_clock::now(); // Record end time
    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
                       .count(); // Calculate duration

    if (result != 0) {
        std::cerr << "Error: Python script execution failed!" << std::endl;
        return -1; // Return -1 to indicate an error
    }

    std::cout << "Python script " << scriptPath << " executed successfully in " << duration
              << " ms." << std::endl;
    return duration;
}


bool finished_transmission(std::vector<NodesIps> nodes_ips, std::vector<Clients_Models>& clients_info)
{
    bool finished = true;
    bool clients_selected = false;

    for (auto c : clients_info) {
        if (c.selected) {
            clients_selected = true;
            bool client_finished = false;
            auto node_id = c.node->GetId();
            Ipv4Address node_ip;

            for (auto nip : nodes_ips) {
                if (nip.node_id == node_id) {
                    node_ip = nip.ip;
                    // LOG("considering client " << node_ip);
                }
            }

            for (auto stream_end_time : endOfStreamTimes) {
                // LOG(stream_end_time.first << node_ip);
                if (stream_end_time.first == node_ip) {
                    // LOG(stream_end_time.first << " " << node_ip);
                    client_finished = true;
                }
            }

            if (!client_finished) {
                finished = false;
            }
        }

        // return true;
    }

    if (finished && clients_selected) {
        LOG("round finished at " << Simulator::Now().GetSeconds() << " seconds.");
        return true;
    } else {
        return false;
    }
}

void network_info(Ptr<FlowMonitor> monitor)
{
    static double lastTotalRxBytes = 0;
    static double lastTotalTxBytes = 0;
    static double lastTime = 0;
    Simulator::Schedule(Seconds(1), &network_info, monitor);
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();
    double totalRxBytes = 0;
    double totalTxBytes = 0;

    for (auto i = stats.begin(); i != stats.end(); ++i) {
        totalRxBytes += i->second.rxBytes;
        totalTxBytes += i->second.txBytes;
    }

    double currentTime = Simulator::Now().GetSeconds();
    double timeDiff = currentTime - lastTime;
    double instantThroughput = (totalRxBytes - lastTotalRxBytes) * 8.0 / timeDiff / 1000 / 1000;
    double instantTxThroughput = (totalTxBytes - lastTotalTxBytes) * 8.0 / timeDiff / 1000 / 1000;
    LOG(currentTime << "s: Instant Network Throughput: " << instantThroughput << " Mbps");
    LOG(currentTime << "s: Instant Tx Throughput: " << instantTxThroughput << " Mbps");
    lastTotalRxBytes = totalRxBytes;
    lastTotalTxBytes = totalTxBytes;
    lastTime = currentTime;
}

void round_cleanup()
{
    endOfStreamTimes.clear();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        auto n_apps = ueNodes.Get(i)->GetNApplications();

        for (uint32_t j = 0; j < n_apps; j++) {
            ueNodes.Get(i)->GetApplication(j)->SetStopTime(Simulator::Now());
            auto app = DynamicCast<MyApp>(ueNodes.Get(i)->GetApplication(j));
            app->StopApplication();
            app->Dispose();
            app = nullptr;
            // ->
        }
    }

    for (uint32_t i = 0; i < remoteHostContainer.GetN(); i++) {
        auto n_apps = remoteHostContainer.Get(i)->GetNApplications();

        for (uint32_t j = 0; j < n_apps; j++) {
            remoteHostContainer.Get(i)->GetApplication(j)->SetStopTime(Simulator::Now());
        }
    }
}