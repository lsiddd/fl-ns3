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
int number_of_ues = 10;
int number_of_enbs = 1;
int n_participaping_clients = 5;
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
    virtual ~MyApp();
    void Setup(Ptr<Socket> socket,
               Address address,
               uint32_t packetSize,
               uint32_t nPackets,
               DataRate dataRate);

  private:
    virtual void StartApplication(void);
    virtual void StopApplication(void);
    void ScheduleTx(void);
    void SendPacket(void);

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    uint32_t m_nPackets;
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running;
    uint32_t m_packetsSent;
    Time m_start_time;
};

// MyApp class implementation
MyApp::MyApp()
    : m_socket(0),
      m_peer(),
      m_packetSize(0),
      m_nPackets(0),
      m_dataRate(0),
      m_sendEvent(),
      m_running(false),
      m_packetsSent(0)
{
}

MyApp::~MyApp()
{
    m_socket = 0;
}

void
MyApp::Setup(Ptr<Socket> socket,
             Address address,
             uint32_t packetSize,
             uint32_t nPackets,
             DataRate dataRate)
{
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_start_time = Simulator::Now();
}

void
MyApp::StartApplication(void)
{
    m_running = true;
    m_packetsSent = 0;
    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
}

void
MyApp::StopApplication(void)
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

void
MyApp::SendPacket(void)
{
    Ptr<Packet> packet = Create<Packet>(m_packetSize);
    if (m_packetsSent + 1 == m_nPackets)
    {
        m_socket->Send(data_fin, writeSize, 0);
    }
    else
    {
        m_socket->Send(data, writeSize, 0);
    }

    if (++m_packetsSent < m_nPackets)
    {
        ScheduleTx();
    }
    else
    {
        Time stop_time = Simulator::Now();
    }
}

void
MyApp::ScheduleTx(void)
{
    if (m_running)
    {
        Time tNext(Seconds(m_packetSize * 8 / static_cast<double>(m_dataRate.GetBitRate())));
        m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
    }
}

// Callback and helper functions
void
RxCallback(std::string path, ns3::Ptr<const ns3::Packet> packet, const ns3::Address& from)
{
    uint32_t packetSize = packet->GetSize();
    std::vector<uint8_t> b(packetSize);
    packet->CopyData(b.data(), packetSize);
    std::string dataAsString(reinterpret_cast<char*>(b.data()), packetSize);

    ns3::InetSocketAddress address = ns3::InetSocketAddress::ConvertFrom(from);
    ns3::Ipv4Address senderIp = address.GetIpv4();

    if (dataAsString.find('b') != std::string::npos)
    {
        double receiveTime = ns3::Simulator::Now().GetSeconds();
        std::cout << "Stream ending signal ('b') received from " << senderIp << " at time "
                  << receiveTime << " seconds." << std::endl;
        endOfStreamTimes[senderIp] = receiveTime;
    }
}

void
sendstream(Ptr<Node> sending_node, Ptr<Node> receiving_node, int size)
{
    port++;
    int n_bytes = 1040;
    int n_packets = size / 1040;

    Ipv4Address receiving_address;
    Ptr<Ipv4> ipv4 = receiving_node->GetObject<Ipv4>();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0);
    Ipv4Address ipAddr = iaddr.GetLocal();

    LOG(Simulator::Now().GetSeconds()
        << "s: Starting stream from node " << sending_node->GetId() << " to node "
        << receiving_node->GetId() << " " << size << " bytes.");

    Address sinkAddress(InetSocketAddress(ipAddr, port));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory",
                                      InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receiving_node);
    sinkApps.Start(Simulator::Now() + Seconds(0.1));

    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(sending_node, TcpSocketFactory::GetTypeId());
    Ptr<MyApp> app = CreateObject<MyApp>();
    app->Setup(ns3TcpSocket, sinkAddress, n_bytes, n_packets, DataRate("1Mb/s"));
    sending_node->AddApplication(app);

    app->SetStartTime(Simulator::Now() + Seconds(0.5));
    app->SetStopTime(Seconds(simStopTime));

    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback(&RxCallback));
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

// Utils functions ===================================
std::streamsize
getFileSize(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return -1;
    }
    return file.tellg();
}

// Run script and measure time function
int64_t
runScriptAndMeasureTime(const std::string& scriptPath)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::string command = "python3 " + scriptPath + " > /dev/null 2>&1";
    int result = system(command.c_str());
    auto end = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (result != 0)
    {
        std::cerr << "Error: Python script execution failed!" << std::endl;
        return -1;
    }

    std::cout << "Python script " << scriptPath << " executed successfully in " << duration << " ms"
              << std::endl;
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

    bool dummy = false;

    if (dummy)
    {
        for (uint32_t i = 0; i < ueNodes.GetN(); i++)
        {
            int training_time = 5000;
            int bytes = 10000;
            clients_info.push_back(Clients_Models(ueNodes.Get(i), training_time, bytes));
        }
        return clients_info;
    }

    bool training_parallel = true;
    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        if (training_parallel) {
            
        // Using async to launch the tasks in parallel
        futures.push_back(std::async(std::launch::async, [i]() {
            std::stringstream cmd;
            cmd << "scratch/client.py --model models/" << ueNodes.Get(i) << "_model.h5 --epochs 5 --n_clients "
                << ueNodes.GetN() << " --id " << i;
            LOG(cmd.str());
            int training_time = runScriptAndMeasureTime(cmd.str()) / ueNodes.GetN();

            // Clear the stringstream and prepare for the next operation
            cmd.str(std::string());
            cmd << "models/" << ueNodes.Get(i) << "_model.tflite";
            int bytes = getFileSize(cmd.str());

            // Return the result as a pair of training time and bytes
            return std::make_pair(training_time, bytes);
        }));
        }
        else {
            for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
                std::stringstream cmd;
                cmd << "scratch/client.py --model models/" << ueNodes.Get(i) << "_model.h5 --epochs 1 --n_clients "
                    << ueNodes.GetN() << " --id " << i;
                LOG(cmd.str());
                int training_time = runScriptAndMeasureTime(cmd.str());

                // Clear the stringstream and prepare for the next operation
                cmd.str(std::string());
                cmd << "models/" << ueNodes.Get(i) << "_model.tflite";
                int bytes = getFileSize(cmd.str());

                clients_info.push_back(Clients_Models(ueNodes.Get(i), training_time, bytes));
            }
            return clients_info;
        }
    }

    // Collect results after all futures have finished executing
    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        auto result = futures[i].get();  // Wait for each future to complete and get the result
        clients_info.push_back(Clients_Models(ueNodes.Get(i), result.first, result.second));
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
                                &sendstream,
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
    static double lastTime = 0;
    Simulator::Schedule(Seconds(1), &network_info, monitor);

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double totalRxBytes = 0;
    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        totalRxBytes += i->second.rxBytes;
    }

    double currentTime = Simulator::Now().GetSeconds();
    double timeDiff = currentTime - lastTime;
    double instantThroughput = (totalRxBytes - lastTotalRxBytes) * 8.0 / timeDiff / 1000 / 1000;

    LOG(currentTime << "s: Instant Network Throughput: " << instantThroughput << " Mbps");

    lastTotalRxBytes = totalRxBytes;
    lastTime = currentTime;
}

// Main function
int
main(int argc, char* argv[])
{
    // Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    // Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    // Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize", UintegerValue(10 * 1024 * 1024));
    // Config::SetDefault("ns3::TcpL4Protocol::SocketType", TypeIdValue(TcpCubic::GetTypeId()));
    // Config::SetDefault("ns3::TcpSocketBase::MinRto", TimeValue(MilliSeconds(200)));
    // Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout", TimeValue(Seconds(0.2)));
    // Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
    // Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    // Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(131072 * 100));
    // Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(131072 * 100));
    // Config::SetDefault("ns3::PhasedArrayModel::AntennaElement",
    //                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    // LogComponentEnable("MmWaveLteRrcProtocolReal", LOG_LEVEL_ALL);
    // LogComponentEnable("mmWaveRrcProtocolIdeal", LOG_LEVEL_ALL);
    // LogComponentEnable("MmWaveUeNetDevice", LOG_LEVEL_ALL);
    Config::SetDefault("ns3::ComponentCarrier::UlBandwidth", UintegerValue(50));
    Config::SetDefault("ns3::ComponentCarrier::PrimaryCarrier", BooleanValue(true));
        Config::SetDefault("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue(false));
    Config::SetDefault("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue(false));
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
    mmwaveHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    mmwaveHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(0));
    mmwaveHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(256)));

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

    // std::string traceFile = "campus.ns_movements";
    // Ns2MobilityHelper ns2 = Ns2MobilityHelper(traceFile);
    // ns2.Install(ueNodes.Begin(), ueNodes.End());
    uemobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    uemobility.SetPositionAllocator(uePositionAlloc);
    uemobility.Install(ueNodes);

    enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbmobility.SetPositionAllocator(enbPositionAlloc);
    enbmobility.Install(enbNodes);
    enbmobility.Install(pgw);
    enbmobility.Install(remoteHost);

    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(43.0));
    Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(20.0));
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