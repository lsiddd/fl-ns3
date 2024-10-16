#define LOG(x) std::cout << x << std::endl

#include "ns3/flow-monitor-module.h"
#include "ns3/applications-module.h"
#include "ns3/command-line.h"
#include "ns3/config-store-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"

#include <chrono>   // For measuring execution time
#include <cstdlib>  // For system() call
#include <fstream>  // For file handling
#include <iostream>
#include <map>
#include <random>
#include <vector>

using namespace ns3;
using namespace mmwave;

NS_LOG_COMPONENT_DEFINE("SidSimulation");

/// Write size.
static const uint32_t writeSize = 2500;
/// Data to be written.
uint8_t data[writeSize] = {'g'};
uint8_t data_fin[writeSize] = {'b'};

std::random_device dev;
std::mt19937 rng(dev());

double simStopTime = 1000;
int number_of_ues = 10;
int number_of_enbs = 10;
int scenario_size = 1000;
std::uniform_int_distribution<std::mt19937::result_type> dist(0, scenario_size);

FlowMonitorHelper flowmon;

NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;

Ipv4Address remoteHostAddr;

std::map<Ipv4Address, double> endOfStreamTimes;

class MyApp : public Application {
   public:
    MyApp();
    virtual ~MyApp();

    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
               uint32_t nPackets, DataRate dataRate);

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

MyApp::MyApp()
    : m_socket(0),
      m_peer(),
      m_packetSize(0),
      m_nPackets(0),
      m_dataRate(0),
      m_sendEvent(),
      m_running(false),
      m_packetsSent(0) {}

MyApp::~MyApp() { m_socket = 0; }

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
                  uint32_t nPackets, DataRate dataRate) {
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_start_time = Simulator::Now();
}

void MyApp::StartApplication(void) {
    m_running = true;
    m_packetsSent = 0;
    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
}

void MyApp::StopApplication(void) {
    m_running = false;

    if (m_sendEvent.IsPending()) {
        Simulator::Cancel(m_sendEvent);
    }

    if (m_socket) {
        m_socket->Close();
    }
}

void MyApp::SendPacket(void) {
    Ptr<Packet> packet = Create<Packet>(m_packetSize);

    // If it's the last packet to send, send 'b'
    if (m_packetsSent + 1 == m_nPackets) {
        m_socket->Send(data_fin, writeSize, 0);  // Sending the 'b' data
    } else {
        m_socket->Send(data, writeSize, 0);  // Sending normal data
    }

    if (++m_packetsSent < m_nPackets) {
        ScheduleTx();
    } else {
        Time stop_time = Simulator::Now();
        // LOG("Application lasted " << (stop_time -
        // m_start_time).GetSeconds()); LOG("Finished Transmission.");
    }
}

void MyApp::ScheduleTx(void) {
    if (m_running) {
        Time tNext(Seconds(m_packetSize * 8 /
                           static_cast<double>(m_dataRate.GetBitRate())));
        m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
    }
}

void RxCallback(std::string path, ns3::Ptr<const ns3::Packet> packet,
                const ns3::Address& from) {
    uint32_t packetSize = packet->GetSize();
    std::vector<uint8_t> b(packetSize);  // Use std::vector instead of VLA
    packet->CopyData(b.data(), packetSize);
    std::string dataAsString(reinterpret_cast<char*>(b.data()), packetSize);

    // Get the sender's IP address
    ns3::InetSocketAddress address = ns3::InetSocketAddress::ConvertFrom(from);
    ns3::Ipv4Address senderIp = address.GetIpv4();

    // Check if the received data contains 'b' (end of stream signal)
    if (dataAsString.find('b') != std::string::npos) {
        double receiveTime = ns3::Simulator::Now().GetSeconds();

        // Log the reception time and store it in the map
        std::cout << "Stream ending signal ('b') received from " << senderIp
                  << " at time " << receiveTime << " seconds." << std::endl;

        // Store in the map
        endOfStreamTimes[senderIp] = receiveTime;
    }
}

uint16_t port = 20000;

void sendstream(Ptr<Node> sending_node, Ptr<Node> receiving_node, int size) {
    // Install and start applications on UEs and remote host
    port++;

    int n_bytes = 1040;
    int n_packets = size / 1040;

    // get the ip address of the receiving device
    Ipv4Address receiving_address;
    Ptr<Ipv4> ipv4 = receiving_node->GetObject<Ipv4>();
    Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0);
    Ipv4Address ipAddr = iaddr.GetLocal();

    // log start of the stream
    LOG( Simulator::Now().GetSeconds() <<  "s: Starting stream from node " << sending_node->GetId() << " to node "
                                     << receiving_node->GetId() << " " << size << " bytes.");

    // create the packet sink in the specified ip and port of the receiving node
    Address sinkAddress(InetSocketAddress(ipAddr, port));
    PacketSinkHelper packetSinkHelper(
        "ns3::TcpSocketFactory",
        InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receiving_node);
    sinkApps.Start(Simulator::Now() + Seconds(0.1));

    // create the tcp socket that will receive the stream
    Ptr<Socket> ns3TcpSocket =
        Socket::CreateSocket(sending_node, TcpSocketFactory::GetTypeId());

    // call the setup of the application on the specified socket
    Ptr<MyApp> app = CreateObject<MyApp>();
    app->Setup(ns3TcpSocket, sinkAddress, n_bytes, n_packets,
               DataRate("1Mb/s"));
    sending_node->AddApplication(app);

    app->SetStartTime(Simulator::Now() + Seconds(0.5));
    app->SetStopTime(Seconds(simStopTime));

    Config::Connect("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx",
                    MakeCallback(&RxCallback));
}

int64_t runScriptAndMeasureTime(const std::string& scriptPath) {
    auto start = std::chrono::high_resolution_clock::now();
    std::string command = "python3 " + scriptPath + " > /dev/null 2>&1";;
    int result = system(command.c_str());
    auto end = std::chrono::high_resolution_clock::now();
    int64_t duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();

    if (result != 0) {
        std::cerr << "Error: Python script execution failed!" << std::endl;
        return -1;
    }

    std::cout << "Python script " << scriptPath << " executed successfully in "
              << duration << " ms" << std::endl;
    return duration;
}

void NotifyConnectionEstablishedUe(std::string context, uint64_t imsi,
                                   uint16_t cellid, uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds " << " UE IMSI "
              << imsi << ": connected to CellId " << cellid << " with RNTI "
              << rnti << std::endl;
}

void NotifyHandoverStartUe(std::string context, uint64_t imsi, uint16_t cellid,
                           uint16_t rnti, uint16_t targetCellId) {
    std::cout << Simulator::Now().GetSeconds() << " seconds " << " UE IMSI "
              << imsi << ": previously connected to CellId " << cellid
              << " with RNTI " << rnti << ", doing handover to CellId "
              << targetCellId << std::endl;
}

void NotifyHandoverEndOkUe(std::string context, uint64_t imsi, uint16_t cellid,
                           uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds " << " UE IMSI "
              << imsi << ": successful handover to CellId " << cellid
              << " with RNTI " << rnti << std::endl;
}

void NotifyConnectionEstablishedEnb(std::string context, uint64_t imsi,
                                    uint16_t cellid, uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds " << " eNB CellId "
              << cellid << ": successful connection of UE with IMSI " << imsi
              << " RNTI " << rnti << std::endl;
}

void NotifyHandoverStartEnb(std::string context, uint64_t imsi, uint16_t cellid,
                            uint16_t rnti, uint16_t targetCellId) {
    std::cout << Simulator::Now().GetSeconds() << " seconds " << " eNB CellId "
              << cellid << ": start handover of UE with IMSI " << imsi
              << " RNTI " << rnti << " to CellId " << targetCellId << std::endl;
}

void NotifyHandoverEndOkEnb(std::string context, uint64_t imsi, uint16_t cellid,
                            uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds " << " eNB CellId "
              << cellid << ": completed handover of UE with IMSI " << imsi
              << " RNTI " << rnti << std::endl;
}

std::map<Ptr<Node>, int> node_to_bytes;
std::map<Ptr<Node>, int> training_time;

std::streamsize getFileSize(const std::string& filename) {
    std::ifstream file(
        filename,
        std::ios::binary | std::ios::ate);  // Open file in binary mode and set
                                            // the position to the end
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return -1;
    }
    return file.tellg();  // Get the current position, which is the file size
}

void federated_round() {

}

void client_selection() {}

void aggregation() {}

void train_clients() {}

void send_models_to_server() {}

void send_models_to_devices() {}



void manager() {
    LOG( "=================== " << Simulator::Now().GetSeconds() << " seconds.");
    // Simulator::Schedule(Seconds(1), &manager);

    // local training
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        std::stringstream cmd;
        cmd << "scratch/client.py --model " << ueNodes.Get(i)
            << "_model.h5 --epochs 1 --n_clients " << ueNodes.GetN() << " --id "
            << i;
        LOG(cmd.str());
        training_time[ueNodes.Get(i)] = runScriptAndMeasureTime(cmd.str());

        cmd.str(std::string());

        cmd << ueNodes.Get(i) << "_model.h5";
        node_to_bytes[ueNodes.Get(i)] = getFileSize(cmd.str());
        // getchar();
    }

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Simulator::Schedule(MilliSeconds(training_time[ueNodes.Get(i)]),
                            &sendstream, ueNodes.Get(i),
                            remoteHostContainer.Get(0),
                            node_to_bytes[ueNodes.Get(i)]);
    }

    runScriptAndMeasureTime("scratch/server.py");
}

void network_info(Ptr<FlowMonitor> monitor) {
    static double lastTotalRxBytes = 0;  // Store the total bytes received from the last call
    static double lastTime = 0;  // Store the time of the last call

    Simulator::Schedule(Seconds(1), &network_info, monitor);

    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double totalRxBytes = 0;

    for (auto i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);

        // std::cout << "Flow " << i->first << " (" << t.sourceAddress << " -> "
        //           << t.destinationAddress << ")\n";
        // std::cout << "  Tx Packets: " << i->second.txPackets << "\n";
        // std::cout << "  Tx Bytes:   " << i->second.txBytes << "\n";
        // std::cout << "  TxOffered:  "
        //           << i->second.txBytes * 8.0 / Simulator::Now().GetSeconds() / 1000 / 1000
        //           << " Mbps\n";
        // std::cout << "  Rx Packets: " << i->second.rxPackets << "\n";
        // std::cout << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        // std::cout << "  Throughput: "
        //           << i->second.rxBytes * 8.0 / Simulator::Now().GetSeconds() / 1000 / 1000
        //           << " Mbps\n";

        // Add the received bytes to the total
        totalRxBytes += i->second.rxBytes;
    }

    // Get the current simulation time
    double currentTime = Simulator::Now().GetSeconds();

    // Calculate the time difference since the last call
    double timeDiff = currentTime - lastTime;

    // Calculate the instant throughput (difference in received bytes / time difference)
    double instantThroughput = (totalRxBytes - lastTotalRxBytes) * 8.0 / timeDiff / 1000 / 1000;

    // Log the instant throughput
    LOG(currentTime << "s: Instant Network Throughput: " << instantThroughput << " Mbps");

    // Update the lastTotalRxBytes and lastTime for the next call
    lastTotalRxBytes = totalRxBytes;
    lastTime = currentTime;
}


int main(int argc, char* argv[]) {
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize",
                       UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize",
                       UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize",
                       UintegerValue(10 * 1024 * 1024));
    Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                       TypeIdValue(TcpCubic::GetTypeId()));
    Config::SetDefault("ns3::TcpSocketBase::MinRto",
                       TimeValue(MilliSeconds(200)));
    Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout",
                       TimeValue(Seconds(0.2)));
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(2500));
    Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
    Config::SetDefault("ns3::TcpSocket::SndBufSize",
                       UintegerValue(131072 * 100));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize",
                       UintegerValue(131072 * 100));
    Config::SetDefault("ns3::PhasedArrayModel::AntennaElement",
                       PointerValue(CreateObject<IsotropicAntennaModel>()));
    // Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(1));
    // Command line arguments
    CommandLine cmd;
    cmd.Parse(argc, argv);

    // Ptr<MmWaveHelper> mmwaveHelper = CreateObject<MmWaveHelper>();
    // mmwaveHelper->SetSchedulerType("ns3::MmWaveFlexTtiMacScheduler");
    // Ptr<MmWavePointToPointEpcHelper> epcHelper =
    // CreateObject<MmWavePointToPointEpcHelper>();
    // mmwaveHelper->SetEpcHelper(epcHelper);

    Ptr<LteHelper> mmwaveHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper =
        CreateObject<PointToPointEpcHelper>();
    mmwaveHelper->SetEpcHelper(epcHelper);

    mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");

    // set up handover
    mmwaveHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    mmwaveHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(3));
    mmwaveHelper->SetHandoverAlgorithmAttribute("TimeToTrigger",
                                                TimeValue(MilliSeconds(256)));

    ConfigStore inputConfig;
    inputConfig.ConfigureDefaults();

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // Create a single RemoteHost
    // NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // Create the Internet
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    // interface 0 is localhost, 1 is the p2p device
    remoteHostAddr = internetIpIfaces.GetAddress(1);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                               Ipv4Mask("255.0.0.0"), 1);

    enbNodes.Create(number_of_enbs);
    ueNodes.Create(number_of_ues);

    // Install Mobility Model
    MobilityHelper enbmobility;
    Ptr<ListPositionAllocator> enbPositionAlloc =
        CreateObject<ListPositionAllocator>();

    MobilityHelper uemobility;
    Ptr<ListPositionAllocator> uePositionAlloc =
        CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        uePositionAlloc->Add(Vector(dist(rng), dist(rng), dist(rng)));
    }
    for (uint32_t i = 0; i < enbNodes.GetN(); i++) {
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

    // Install LTE Devices to the nodes
    NetDeviceContainer enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);
    mmwaveHelper->EnableTraces();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(
            epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    Ptr<FlowMonitor> monitor = flowmon.InstallAll();    // CheckThroughput();

    Simulator::Schedule(Seconds(1), &manager);
    Simulator::Schedule(Seconds(1), &network_info, monitor);

    // monitor->SerializeToXmlFile("simulation.flowmon", false, false);

    AnimationInterface anim("mmwave-animation.xml");  // Mandatory
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        anim.UpdateNodeDescription(ueNodes.Get(i), "UE");  // Optional
        anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0);   // Optional
    }
    for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
        anim.UpdateNodeDescription(enbNodes.Get(i), "ENB");  // Optional
        anim.UpdateNodeColor(enbNodes.Get(i), 0, 255, 0);    // Optional
    }
    anim.UpdateNodeDescription(remoteHost, "RH");  // Optional
    anim.UpdateNodeColor(remoteHost, 0, 0, 255);   // Optional
    anim.UpdateNodeDescription(pgw, "pgw");        // Optional
    anim.UpdateNodeColor(pgw, 0, 0, 255);          // Optional

    // connect custom trace sinks for RRC connection establishment and handover
    // notification
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

    // Print out when each address received the end of stream signal
    std::cout << "End of stream times per IP address:" << std::endl;
    for (const auto& entry : endOfStreamTimes) {
        std::cout << "IP Address: " << entry.first
                  << " received the end signal at time: " << entry.second
                  << " seconds." << std::endl;
    }

    Simulator::Destroy();

    return 0;
}
