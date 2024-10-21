#define LOG(x) std::cout << x << std::endl

#include "MyApp.h"
#include "client_types.h"
#include "json.hpp"
#include "notifications.h"
#include "utils.h"

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
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <vector>
#include <unistd.h>

using namespace ns3;
using namespace mmwave;
using json = nlohmann::json;

NS_LOG_COMPONENT_DEFINE("Simulation");

// Global variables and constants
double simStopTime = 3600;
int number_of_ues = 2;
int number_of_enbs = 2;
int n_participaping_clients = number_of_ues / 2;
int scenario_size = 1000;

NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;
NetDeviceContainer enbDevs;
NetDeviceContainer ueDevs;
Ipv4Address remoteHostAddr;

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(0, scenario_size);
FlowMonitorHelper flowmon;

std::map<Ipv4Address, double> endOfStreamTimes;
std::map<Ptr<Node>, int> node_to_bytes, training_time;
std::map<uint16_t, std::map<uint16_t, double>> sinr_ue;
std::map<uint16_t, std::map<uint16_t, double>> rsrp_ue;

static bool round_finished = true;
static int round_number = 0;
std::vector<NodesIps> nodes_ips;
std::vector<Clients_Models> clients_info;
std::vector<Clients_Models> selected_clients;
Time timeout = Seconds(60);

std::vector<Clients_Models>
train_clients()
{
    std::vector<Clients_Models> clients_info;
    std::vector<std::future<std::pair<int, int>>> futures; // Store futures for parallel execution

    LOG("=================== " << Simulator::Now().GetSeconds() << " seconds.");

    bool dummy = false;

    // If dummy mode is enabled, return mock data
    if (dummy)
    {
        const int training_time = 5000; // Constant training time for dummy mode
        const int bytes = 1000000;      // Constant bytes for dummy mode

        for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
        {
            Ptr<NetDevice> ueDevice = ueDevs.Get(i);
            auto rnti = ueDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetRnti();
            auto cellId = ueDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetCellId();
            // Get RSRP and SINR from the global maps
            double rsrp = rsrp_ue[cellId][rnti];
            double sinr = sinr_ue[cellId][rnti];

            clients_info.emplace_back(ueNodes.Get(i), training_time, bytes, rsrp, sinr);
        }
        return clients_info;
    }

    // Sequential training (non-parallel)
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        std::stringstream cmd;

        cmd << "curl -X POST \"http://127.0.0.1:8000/train\"  -H \"Content-Type: "
               "application/json\" -d '{\"n_clients\": "
            << ueNodes.GetN() << ", \"client_id\": " << i
            << ", \"epochs\": 1, "
               "\"model\": \"models/"
            << ueNodes.Get(i) << ".keras\", \"top_n\": 3}'";

        // cmd << "scratch/client.py --model models/" << ueNodes.Get(i)
        //     << "_model.keras --epochs 1 --n_clients " << ueNodes.GetN() << " --id " << i;

        LOG(cmd.str()); // Log the command being executed

        // int training_time = runScriptAndMeasureTime(cmd.str());
        int result = system(cmd.str().c_str());
    }

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        std::stringstream finish_file;
        std::stringstream cmd;
        finish_file << "models/" << ueNodes.Get(i) << ".finish";
        if (std::filesystem::exists(finish_file.str()))
        {
            sleep(0.2);//sleeps for 3 second
            LOG("this one has finished");

            // Reset the stringstream to get the model size
            cmd.str(std::string());
            cmd << "models/" << ueNodes.Get(i) << ".keras";
            int bytes = getFileSize(cmd.str());

            Ptr<NetDevice> ueDevice = ueDevs.Get(i);
            auto rnti = ueDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetRnti();
            auto cellId = ueDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetCellId();
            // Get RSRP and SINR from the global maps
            double rsrp = rsrp_ue[cellId][rnti];
            double sinr = sinr_ue[cellId][rnti];


            std::stringstream json_filename;
            json_filename << "models/" << ueNodes.Get(i) << "_model_sizes.json";
            std::ifstream ifs(json_filename.str());
            json j;
            ifs >> j;
            int training_time = j["duration"];
            LOG(training_time);

            // Store the client information
            clients_info.emplace_back(ueNodes.Get(i), training_time, bytes, rsrp, sinr);

            std::stringstream rm_command;
            rm_command << "rm models/" << ueNodes.Get(i) << ".finish";
            system(rm_command.str().c_str());
        }
        else
        {
            i--;
        }
    }
    // getchar();

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

    // Create a JSON object to store selected client models
    json selected_clients_json;

    for (auto i : numbers)
    {
        LOG(clients_info[i]);
        clients_info[i].selected = true;

        // Add the model filename of the selected client to the JSON object
        std::stringstream model_filename;
        model_filename << "models/" << ueNodes.Get(i) << "_model.keras";
        selected_clients_json["selected_clients"].push_back(model_filename.str());
    }

    // Save the JSON object to a file
    std::ofstream out("selected_clients.json");
    out << std::setw(4) << selected_clients_json << std::endl;
    out.close();

    return clients_info;
}

void
aggregation()
{
    runScriptAndMeasureTime("scratch/server.py");
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
manager()
{
    static Time round_start;

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
    enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);

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

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        // Access the LteUePhy from the UE device
        Ptr<LteUePhy> uePhy = ueDevs.Get(i)->GetObject<LteUeNetDevice>()->GetPhy();

        // Connect trace source to monitor SINR and RSRP
        uePhy->TraceConnectWithoutContext("ReportCurrentCellRsrpSinr",
                                          MakeCallback(&ReportUeSinrRsrp));
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
