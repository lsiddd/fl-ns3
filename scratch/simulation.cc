#define LOG(x) std::cout << x << std::endl

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
#include "json.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <future>
#include <vector>
#include <sstream>


using namespace ns3;
using namespace mmwave;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("Simulation");

// Global variables and constants
static const uint32_t writeSize = 2500;
uint8_t data[writeSize] = {'g'};
uint8_t data_fin[writeSize] = {'b'};
double simStopTime = 3600;
int number_of_ues = 20;
int number_of_enbs = 20;
int n_participaping_clients = number_of_ues / 2;
int scenario_size = 1000;

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(0, scenario_size);
FlowMonitorHelper flowmon;
NodeContainer ueNodes, enbNodes, remoteHostContainer;
Ipv4Address remoteHostAddr;
std::map<Ipv4Address, double> endOfStreamTimes;
uint16_t port = 20000;
std::map<Ptr<Node>, int> node_to_bytes, training_time;

// MyApp class definition
class MyApp : public Application
{
public:
    MyApp();
    virtual ~MyApp() override;
    
    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize, 
               uint32_t nPackets, DataRate dataRate);
    virtual void StopApplication() override;

private:
    virtual void StartApplication() override;
    void ScheduleTx();
    void SendPacket();

    Ptr<Socket> m_socket{nullptr};  // Use nullptr for pointer initialization
    Address m_peer;
    uint32_t m_packetSize{0};
    uint32_t m_nPackets{0};
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running{false};
    uint32_t m_packetsSent{0};
    Time m_startTime;  // Renamed for consistency
};

// MyApp class implementation
MyApp::MyApp() = default;  // Use default constructor

MyApp::~MyApp()
{
    m_socket = nullptr;  // Set to nullptr for clarity
}

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
                  uint32_t nPackets, DataRate dataRate)
{
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_startTime = Simulator::Now();
}

void MyApp::StartApplication()
{
    m_running = true;
    m_packetsSent = 0;
    
    // Check if socket is valid before using it
    if (!m_socket)
    {
        NS_FATAL_ERROR("Socket not initialized");
    }

    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
}

void MyApp::StopApplication()
{
    m_running = false;

    if (m_sendEvent.IsPending())
    {
        Simulator::Cancel(m_sendEvent);
    }

    if (m_socket)
    {
        m_socket->Close();
    }
}

void MyApp::SendPacket()
{
    Ptr<Packet> packet = Create<Packet>(m_packetSize);
    
    // Logic to handle last packet separately
    if (m_packetsSent + 1 == m_nPackets)
    {
        m_socket->Send(data_fin, writeSize, 0);  // Placeholder: data_fin should be defined elsewhere
    }
    else
    {
        m_socket->Send(data, writeSize, 0);  // Placeholder: data should be defined elsewhere
    }

    ++m_packetsSent;

    if (m_packetsSent < m_nPackets)
    {
        ScheduleTx();
    }
    else
    {
        Time stopTime = Simulator::Now();  // Consider logging or using stopTime
    }
}

void MyApp::ScheduleTx()
{
    if (m_running)
    {
        double seconds = static_cast<double>(m_packetSize * 8) / m_dataRate.GetBitRate();
        Time tNext = Seconds(seconds);
        m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
    }
}

// Callback and helper functions
void RxCallback(const std::string path, Ptr<const Packet> packet, const Address& from)
{
    uint32_t packetSize = packet->GetSize();
    std::vector<uint8_t> buffer(packetSize);
    packet->CopyData(buffer.data(), packetSize);
    
    std::string dataAsString(reinterpret_cast<char*>(buffer.data()), packetSize);

    InetSocketAddress address = InetSocketAddress::ConvertFrom(from);
    Ipv4Address senderIp = address.GetIpv4();

    if (dataAsString.find('b') != std::string::npos)
    {
        double receiveTime = Simulator::Now().GetSeconds();
        std::cout << "Stream ending signal ('b') received from " << senderIp 
                  << " at time " << receiveTime << " seconds." << std::endl;
        
        endOfStreamTimes[senderIp] = receiveTime;  // Assuming endOfStreamTimes is declared somewhere
    }
}

void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size)
{
    static uint16_t port = 5000;  // Initialized once and incremented for each call
    port++;

    constexpr int packetSize = 1040;  // Defined as a constant for better readability
    int nPackets = size / packetSize; // Calculate the number of packets

    // Get receiving node's IPv4 address
    Ptr<Ipv4> ipv4 = receivingNode->GetObject<Ipv4>();
    if (!ipv4)
    {
        NS_FATAL_ERROR("No Ipv4 object found on receiving node");
    }

    Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0);  // Assume the second interface
    Ipv4Address ipAddr = iaddr.GetLocal();

    // Logging the stream initiation
    LOG(Simulator::Now().GetSeconds()
        << "s: Starting stream from node " << sendingNode->GetId() 
        << " to node " << receivingNode->GetId() << ", " 
        << size << " bytes.");

    // Create and configure the sink (receiver)
    Address sinkAddress(InetSocketAddress(ipAddr, port));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", 
                                      InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receivingNode);
    sinkApps.Start(Seconds(0.1));

    // Create the sending socket
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(sendingNode, TcpSocketFactory::GetTypeId());
    if (!ns3TcpSocket)
    {
        NS_FATAL_ERROR("Failed to create TCP socket on sending node");
    }

    // Create and configure the application to send packets
    Ptr<MyApp> app = CreateObject<MyApp>();
    app->Setup(ns3TcpSocket, sinkAddress, packetSize, nPackets, DataRate("10Mb/s"));
    sendingNode->AddApplication(app);

    // Schedule the start and stop times for the application
    app->SetStartTime(Seconds(0.5));
    // app->SetStopTime(Seconds(simStopTime));  // Assuming simStopTime is declared globally

    // Connect the callback for when packets are received at the sink
    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx", 
                    MakeCallback(&RxCallback));
}

// Event handling functions
void
NotifyConnectionEstablishedUe(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << " seconds UE IMSI " << imsi
              << ": connected to CellId " << cellid << " with RNTI " << rnti << std::endl;
}

void
NotifyHandoverStartUe(std::string context,
                      uint64_t imsi,
                      uint16_t cellid,
                      uint16_t rnti,
                      uint16_t targetCellId)
{
    std::cout << Simulator::Now().GetSeconds() << " seconds UE IMSI " << imsi
              << ": previously connected to CellId " << cellid << ", doing handover to CellId "
              << targetCellId << std::endl;
}

void
NotifyHandoverEndOkUe(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << " seconds UE IMSI " << imsi
              << ": successful handover to CellId " << cellid << " with RNTI " << rnti << std::endl;
}

void
NotifyConnectionEstablishedEnb(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << " seconds eNB CellId " << cellid
              << ": successful connection of UE with IMSI " << imsi << " RNTI " << rnti
              << std::endl;
}

void
NotifyHandoverStartEnb(std::string context,
                       uint64_t imsi,
                       uint16_t cellid,
                       uint16_t rnti,
                       uint16_t targetCellId)
{
    std::cout << Simulator::Now().GetSeconds() << " seconds eNB CellId " << cellid
              << ": start handover of UE with IMSI " << imsi << " RNTI " << rnti << " to CellId "
              << targetCellId << std::endl;
}

void
NotifyHandoverEndOkEnb(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << " seconds eNB CellId " << cellid
              << ": completed handover of UE with IMSI " << imsi << " RNTI " << rnti << std::endl;
}

// Utility Functions ==================================
// Function to get the size of a file
std::streamsize getFileSize(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);  // Open file in binary mode at the end
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file: " << filename << std::endl;
        return -1;  // Return -1 to indicate an error
    }

    return file.tellg();  // Get the file size by reading the position of the end of the file
}

// Function to extract model path without suffix and extension
std::string extractModelPath(const std::string& input) 
{
    const std::string modelFlag = "--model ";
    const std::string extension = ".keras";
    const std::string modelSuffix = "_model";

    size_t modelPos = input.find(modelFlag);
    if (modelPos == std::string::npos) 
    {
        return "";  // Return empty string if "--model" flag is not found
    }

    // Start after the "--model " flag
    size_t start = modelPos + modelFlag.length();
    // Find the position of the next space after the model path
    size_t end = input.find(" ", start);
    if (end == std::string::npos) 
    {
        end = input.length();  // If no space is found, assume the model path goes to the end
    }

    std::string modelPath = input.substr(start, end - start);  // Extract model path

    // Remove the ".keras" extension if it exists
    size_t extensionPos = modelPath.find(extension);
    if (extensionPos != std::string::npos) 
    {
        modelPath = modelPath.substr(0, extensionPos);
    }

    // Remove the "_model" suffix if it exists
    size_t modelSuffixPos = modelPath.rfind(modelSuffix);
    if (modelSuffixPos != std::string::npos) 
    {
        modelPath = modelPath.substr(0, modelSuffixPos);
    }

    return modelPath;
}

// Function to run a Python script and measure its execution time
int64_t runScriptAndMeasureTime(const std::string& scriptPath)
{
    auto startTime = std::chrono::high_resolution_clock::now();  // Record start time

    std::string modelPath = extractModelPath(scriptPath);
    std::string cmdOutputFile = modelPath + ".txt";
    std::string command = "python3 " + scriptPath + " > " + cmdOutputFile + " 2>&1";  // Redirect output to a file

    int result = system(command.c_str());  // Run the Python script
    auto endTime = std::chrono::high_resolution_clock::now();  // Record end time

    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();  // Calculate duration

    if (result != 0)
    {
        std::cerr << "Error: Python script execution failed!" << std::endl;
        return -1;  // Return -1 to indicate an error
    }

    std::cout << "Python script " << scriptPath << " executed successfully in " << duration << " ms." << std::endl;
    return duration;
}


struct Clients_Models
{
    Ptr<Node> node;
    int training_time;
    int node_to_bytes;
    bool selected;

    // Constructor with all three parameters
    Clients_Models(Ptr<Node> n, int t, int b, bool s)
        : node(n),
          training_time(t),
          node_to_bytes(b),
          selected(s)
    {
    }

    // Constructor with two parameters, defaulting 'selected' to false
    Clients_Models(Ptr<Node> n, int t, int b)
        : node(n),
          training_time(t),
          node_to_bytes(b),
          selected(false)
    {
    }

    // Explicitly delete the default constructor
    Clients_Models() = delete;
};

// Overload the '<<' operator for Clients_Models
std::ostream&
operator<<(std::ostream& os, const Clients_Models& model)
{
    os << "Clients_Models { id: " << model.node->GetId()
       << ", training_time: " << model.training_time << ", node_to_bytes: " << model.node_to_bytes
       << ", selected: " << (model.selected ? "true" : "false") << " }";
    return os;
}

struct NodesIps
{
    uint32_t node_id;
    uint32_t index;
    Ipv4Address ip;

    NodesIps(int n, int i, Ipv4Address ia)
        : node_id(n),
          index(i),
          ip(ia)
    {
    }
};

std::vector<NodesIps>
node_to_ips()
{
    std::vector<NodesIps> nodes_ips;
    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        Ipv4Address receiving_address;
        Ptr<Ipv4> ipv4 = ueNodes.Get(i)->GetObject<Ipv4>();
        Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0);
        Ipv4Address ipAddr = iaddr.GetLocal();
        nodes_ips.push_back(NodesIps(ueNodes.Get(i)->GetId(), i, ipAddr));
    }
    return nodes_ips;
}

std::vector<Clients_Models> train_clients()
{
    std::vector<Clients_Models> clients_info;
    std::vector<std::future<std::pair<int, int>>> futures; // Store futures for parallel execution

    LOG("=================== " << Simulator::Now().GetSeconds() << " seconds.");

    bool dummy = true;

    // If dummy mode is enabled, return mock data
    if (dummy)
    {
        const int training_time = 5000;  // Constant training time for dummy mode
        const int bytes = 1000000;       // Constant bytes for dummy mode
        
        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            clients_info.emplace_back(ueNodes.Get(i), training_time, bytes);
        }
        return clients_info;
    }

    bool training_parallel = false;

    // If training is parallel, use async to run training concurrently
    if (training_parallel)
    {
        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            // Launch tasks asynchronously
            futures.push_back(std::async(std::launch::async, [i]() {
                std::stringstream cmd;
                cmd << "scratch/client.py --model models/" << ueNodes.Get(i) 
                    << "_model.keras --epochs 5 --n_clients " << ueNodes.GetN() 
                    << " --id " << i;

                LOG(cmd.str());  // Log the command being executed
                
                int training_time = runScriptAndMeasureTime(cmd.str()) / ueNodes.GetN();

                // Reset the stringstream and generate model path
                cmd.str(std::string());
                cmd << "models/" << ueNodes.Get(i) << "_model.keras";
                int bytes = getFileSize(cmd.str());

                // Return the result as a pair of training time and bytes
                return std::make_pair(training_time, bytes);
            }));
        }

        // Collect results after all futures are finished
        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            auto result = futures[i].get();  // Wait for the result of each future
            clients_info.emplace_back(ueNodes.Get(i), result.first, result.second);
        }
    }
    else
    {
        // Sequential training (non-parallel)
        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            std::stringstream cmd;
            cmd << "scratch/client.py --model models/" << ueNodes.Get(i) 
                << "_model.keras --epochs 1 --n_clients " << ueNodes.GetN() 
                << " --id " << i;

            LOG(cmd.str());  // Log the command being executed

            int training_time = runScriptAndMeasureTime(cmd.str());

            // Reset the stringstream to get the model size
            cmd.str(std::string());
            cmd << "models/" << ueNodes.Get(i) << "_model.keras";
            int bytes = getFileSize(cmd.str());

            // Store the client information
            clients_info.emplace_back(ueNodes.Get(i), training_time, bytes);
        }
    }

    return clients_info;
}


std::vector<Clients_Models>
client_selection(int n, std::vector<Clients_Models> clients_info)
{
    std::vector<uint32_t> selected(ueNodes.GetN(), 0);
    std::vector<int> numbers(ueNodes.GetN()); // Inclusive range [0, N]
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        numbers[i] = i;
    }

    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::shuffle(numbers.begin(), numbers.end(), gen);

    numbers.resize(n);

    for (auto i : numbers)
    {
        clients_info[i].selected = true;
    }
    return clients_info;
}

void
aggregation()
{
    runScriptAndMeasureTime("scratch/server.py");
}

void
federated_round()
{
}

void
send_models_to_server(std::vector<Clients_Models> clients)
{
    for (auto i : clients)
    {
        if (i.selected)
        {
            LOG("Client " << i << " scheduling send model.");
            Simulator::Schedule(MilliSeconds(i.training_time),
                                &sendStream,
                                i.node,
                                remoteHostContainer.Get(0),
                                i.node_to_bytes);
        }
    }
}

void
send_models_to_devices()
{
}

static bool round_finished = true;
static int round_number = 0;

bool
finished_transmission(std::vector<NodesIps> nodes_ips, std::vector<Clients_Models>& clients_info)
{
    bool finished = true;
    bool clients_selected = false;
    for (auto c : clients_info)
    {
        if (c.selected)
        {
            clients_selected = true;
            bool client_finished = false;
            auto node_id = c.node->GetId();
            Ipv4Address node_ip;
            for (auto nip : nodes_ips)
            {
                if (nip.node_id == node_id)
                {
                    node_ip = nip.ip;
                    // LOG("considering client " << node_ip);
                }
            }
            for (auto stream_end_time : endOfStreamTimes)
            {
                // LOG(stream_end_time.first << node_ip);
                if (stream_end_time.first == node_ip)
                {
                    // LOG(stream_end_time.first << " " << node_ip);
                    client_finished = true;
                }
            }
            if (!client_finished)
            {
                finished = false;
            }
        }

        // return true;
    }
    if (finished && clients_selected)
    {
        LOG("round finished");
        return true;
    }
    else
    {
        return false;
    }
}

std::vector<NodesIps> nodes_ips;
std::vector<Clients_Models> clients_info;
std::vector<Clients_Models> selected_clients;
Time timeout = Seconds(60);

void
round_cleanup()
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


void
manager()
{
    static Time round_start;

    // std::ifstream f("fashionmnist_quantized_model_model_sizes.json");
    // json data = json::parse(f);
    // LOG(data);
    // getchar();

    if (Simulator::Now() - round_start > timeout)
    {
        round_finished = true;
        LOG("Round timed out, not all clients were able to send " << endOfStreamTimes.size() << "/"
                                                                  << n_participaping_clients);
    }

    nodes_ips = node_to_ips();
    if (round_finished)
    {
        round_cleanup();
        round_start = Simulator::Now();
        round_number++;
        round_finished = false;

        clients_info = train_clients();
        selected_clients = client_selection(n_participaping_clients, clients_info);

        send_models_to_server(selected_clients);
        aggregation();
    }
    else
    {
        LOG("round not finished");
    }

    round_finished = finished_transmission(nodes_ips, selected_clients);

    Simulator::Schedule(Seconds(1), &manager);
}

void
network_info(Ptr<FlowMonitor> monitor)
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
    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
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

// Main function
int
main(int argc, char* argv[])
{


    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024 * 10));
    Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
    Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(MilliSeconds(200)));
    Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout", TimeValue(Seconds(2)));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(131072 * 100 * 10));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(131072 * 100 * 10));
    // Config::SetDefault("ns3::PhasedArrayModel::AntennaElement",
    //                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    // LogComponentEnable("MmWaveLteRrcProtocolReal", LOG_LEVEL_ALL);
    // LogComponentEnable("mmWaveRrcProtocolIdeal", LOG_LEVEL_ALL);
    // LogComponentEnable("MmWaveUeNetDevice", LOG_LEVEL_ALL);
    Config::SetDefault("ns3::ComponentCarrier::UlBandwidth", UintegerValue(50));
    Config::SetDefault("ns3::ComponentCarrier::PrimaryCarrier", BooleanValue(true));
        Config::SetDefault("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::LteHelper::UsePdschForCqiGeneration", BooleanValue(true));

    // Uplink Power Control
    Config::SetDefault("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue(true));
    Config::SetDefault("ns3::LteUePowerControl::ClosedLoop", BooleanValue(true));
    Config::SetDefault("ns3::LteUePowerControl::AccumulationEnabled", BooleanValue(false));

    CommandLine cmd;
    cmd.Parse(argc, argv);

    // Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper>();
    // mmwaveHelper->SetSchedulerType("ns3::MmWaveFlexTtiMacScheduler");
    // Ptr<MmWavePointToPointEpcHelper> epcHelper =
    // CreateObject<MmWavePointToPointEpcHelper>();
    // mmwaveHelper->SetEpcHelper(epcHelper);
    // Config::SetDefault("ns3::LteEnbRrc::SecondaryCellHandoverMode",
    //                        EnumValue(LteEnbRrc::THRESHOLD));



    Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    mmwaveHelper->SetEpcHelper(epcHelper);
    mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    // mmwaveHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    // mmwaveHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(3));
    // mmwaveHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(256)));
    mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    mmwaveHelper->SetHandoverAlgorithmAttribute("ServingCellThreshold", UintegerValue(30));
    mmwaveHelper->SetHandoverAlgorithmAttribute("NeighbourCellOffset", UintegerValue(1));

    ConfigStore inputConfig;
    inputConfig.ConfigureDefaults();

    Ptr<Node> pgw = epcHelper->GetPgwNode();
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
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
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    enbNodes.Create(number_of_enbs);
    ueNodes.Create(number_of_ues);

    MobilityHelper enbmobility;
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    MobilityHelper uemobility;
    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        uePositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }
    for (uint32_t i = 0; i < enbNodes.GetN(); i++)
    {
        enbPositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }

    std::string traceFile = "campus.ns_movements";
    Ns2MobilityHelper ns2 = Ns2MobilityHelper(traceFile);
    ns2.Install(ueNodes.Begin(), ueNodes.End());
    // uemobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    // uemobility.SetPositionAllocator(uePositionAlloc);
    // uemobility.Install(ueNodes);

    enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbmobility.SetPositionAllocator(enbPositionAlloc);
    enbmobility.Install(enbNodes);
    enbmobility.Install(pgw);
    enbmobility.Install(remoteHost);

    // Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(43.0));
    // Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(20.0));
    NetDeviceContainer enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    mmwaveHelper->AddX2Interface(enbNodes);
    mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);
    // mmwaveHelper->EnableTraces();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    Simulator::Schedule(Seconds(1), &manager);
    Simulator::Schedule(Seconds(1), &network_info, monitor);

    // AnimationInterface anim("mmwave-animation.xml");
    // for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    // {
    //     anim.UpdateNodeDescription(ueNodes.Get(i), "UE");
    //     anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0);
    // }
    // for (uint32_t i = 0; i < enbNodes.GetN(); ++i)
    // {
    //     anim.UpdateNodeDescription(enbNodes.Get(i), "ENB");
    //     anim.UpdateNodeColor(enbNodes.Get(i), 0, 255, 0);
    // }
    // anim.UpdateNodeDescription(remoteHost, "RH");
    // anim.UpdateNodeColor(remoteHost, 0, 0, 255);
    // anim.UpdateNodeDescription(pgw, "pgw");
    // anim.UpdateNodeColor(pgw, 0, 0, 255);

    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkUe));

    Simulator::Stop(Seconds(simStopTime));
    Simulator::Run();

    std::cout << "End of stream times per IP address:" << std::endl;
    for (const auto& entry : endOfStreamTimes)
    {
        std::cout << "IP Address: " << entry.first
                  << " received the end signal at time: " << entry.second << " seconds."
                  << std::endl;
    }

    Simulator::Destroy();
    return 0;
}
