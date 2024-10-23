// Macro for logging
#define LOG(x) std::cout << x << std::endl

// Project-specific headers
#include "MyApp.h"
#include "client_types.h"
#include "notifications.h"
#include "utils.h"

// External library headers
#include "json.hpp"

// NS-3 module headers
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

// Standard Library headers
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <vector>

// Using declarations for convenience
using namespace ns3;
using namespace mmwave;
using json = nlohmann::json;

// Define the simulation logging component
NS_LOG_COMPONENT_DEFINE("Simulation");

// Global constants
static constexpr double simStopTime = 300.0;
static constexpr int number_of_ues = 2;
static constexpr int number_of_enbs = 10;
static constexpr int n_participating_clients = number_of_ues / 2;
static constexpr int scenario_size = 1000;

// Global variables for simulation objects
NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;
NetDeviceContainer enbDevs;
NetDeviceContainer ueDevs;
Ipv4Address remoteHostAddr;

// Random number generation setup
std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(0, scenario_size);

// Flow monitoring helper
FlowMonitorHelper flowmon;

// Data structure for tracking various metrics
std::map<Ipv4Address, double> endOfStreamTimes;
std::map<Ptr<Node>, int> node_to_bytes, training_time;
std::map<uint16_t, std::map<uint16_t, double>> sinr_ue;
std::map<uint16_t, std::map<uint16_t, double>> rsrp_ue;

// Global state variables
static bool round_finished = true;
static int round_number = 0;

// Client-related information
std::vector<NodesIps> nodes_ips;
std::vector<Clients_Models> clients_info;
std::vector<Clients_Models> selected_clients;

// Timeout for certain operations
Time timeout = Seconds(60);

std::vector<Clients_Models>
train_clients()
{
    std::vector<Clients_Models> clients_info;

    LOG("=================== " << Simulator::Now().GetSeconds() << " seconds.");

    bool dummy = false;

    // Helper function to get RSRP and SINR values
    auto get_rsrp_sinr = [](uint32_t nodeIdx) -> std::pair<double, double> {
        Ptr<NetDevice> ueDevice = ueDevs.Get(nodeIdx);
        auto rnti = ueDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetRnti();
        auto cellId = ueDevice->GetObject<LteUeNetDevice>()->GetRrc()->GetCellId();
        double rsrp = rsrp_ue[cellId][rnti];
        double sinr = sinr_ue[cellId][rnti];
        return {rsrp, sinr};
    };

    // Helper function to check if a file exists
    auto file_exists = [](const std::string& filename) -> bool {
        return std::filesystem::exists(filename);
    };

    // Helper function to get file size
    auto get_file_size = [](const std::string& filepath) -> int {
        std::ifstream in(filepath, std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    };

    // Helper function to parse JSON file
    auto parse_json_file = [](const std::string& filepath) -> json {
        std::ifstream ifs(filepath);
        json j;
        ifs >> j;
        return j;
    };

    if (dummy) {
        const int training_time = 5000; // Constant training time for dummy mode
        const int bytes = 1000000;      // Constant bytes for dummy mode

        for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
            auto [rsrp, sinr] = get_rsrp_sinr(i);
            clients_info.emplace_back(ueNodes.Get(i), training_time, bytes, rsrp, sinr);
        }
        return clients_info;
    }

    // Sequential training (non-parallel)
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        std::stringstream cmd;
        cmd << "curl -X POST \"http://127.0.0.1:8182/train\"  -H \"Content-Type: "
            "application/json\" -d '{\"n_clients\": "
            << ueNodes.GetN() << ", \"client_id\": " << i
            << ", \"epochs\": 1, "
            "\"model\": \"models/"
            << ueNodes.Get(i) << ".keras\", \"top_n\": 3}'";

        LOG(cmd.str()); // Log the command being executed

        // Use system() or any alternative to run the script and measure time
        // system(cmd.str().c_str());
        int ret = system(cmd.str().c_str());
        if (ret != 0) {
            LOG("Command failed with return code: " << ret);
        }
    }

    // Collect the training results
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        std::stringstream finish_file;
        finish_file << "models/" << ueNodes.Get(i) << ".finish";

        if (file_exists(finish_file.str())) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Sleep for 200ms
            // LOG("this one has finished");

            std::stringstream model_file;
            model_file << "models/" << ueNodes.Get(i) << ".keras";
            int bytes = get_file_size(model_file.str());

            // Retrieve RSRP and SINR for the current client
            auto [rsrp, sinr] = get_rsrp_sinr(i);

            // Parse the JSON file for training duration
            std::stringstream json_filename;
            json_filename << "models/" << ueNodes.Get(i) << "_model_sizes.json";
            json j = parse_json_file(json_filename.str());
            int training_time = j["duration"];
            // LOG(training_time);

            LOG("\nClient " << i << " finished training after " << training_time
                << " milliseconds");

            // Store the client information
            clients_info.emplace_back(ueNodes.Get(i), training_time, bytes, rsrp, sinr);

            // Clean up the .finish file
            std::stringstream rm_command;
            rm_command << "rm models/" << ueNodes.Get(i) << ".finish";
            // system(rm_command.str().c_str());
            int ret_rm = system(rm_command.str().c_str());
            if (ret_rm != 0) {
                LOG("Failed to remove file, command returned: " << ret_rm);
            }
        } else {
            --i; // Retry if the file isn't ready yet
        }
    }

    return clients_info;
}

std::vector<Clients_Models>
client_selection(int n, std::vector<Clients_Models> clients_info)
{
    std::vector<uint32_t> selected(ueNodes.GetN(), 0);
    std::vector<int> numbers(ueNodes.GetN()); // Inclusive range [0, N]
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        numbers[i] = i;
    }

    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::shuffle(numbers.begin(), numbers.end(), gen);

    numbers.resize(n);

    // Create a JSON object to store selected client models
    json selected_clients_json;

    for (auto i : numbers) {
        LOG(clients_info[i]);
        clients_info[i].selected = true;

        // Add the model filename of the selected client to the JSON object
        std::stringstream model_filename;
        model_filename << "models/" << ueNodes.Get(i) << ".keras";
        selected_clients_json["selected_clients"].push_back(model_filename.str());
    }

    // Save the JSON object to a file
    std::ofstream out("selected_clients.json");
    out << std::setw(4) << selected_clients_json << std::endl;
    out.close();

    return clients_info;
}

void
log_server_evaluation()
{
    // Helper function to parse JSON file with error handling
    auto parse_json_file = [](const std::string& filepath) -> nlohmann::json {
        std::ifstream ifs(filepath);
        nlohmann::json j;

        // Check if file stream is valid (file exists and is readable)
        if (!ifs.is_open())
        {
            LOG("Error: Could not open file " << filepath);
            return {}; // Return an empty JSON object
        }

        try
        {
            ifs >> j; // Attempt to parse the JSON
        } catch (nlohmann::json::parse_error& e)
        {
            LOG("Error: Failed to parse JSON file " << filepath << ". Error: " << e.what());
            return {}; // Return an empty JSON object on parse failure
        }

        return j; // Return the parsed JSON object if successful
    };

    std::string json_file = "evaluation_metrics.json";

    // Call the parse function and handle the result
    nlohmann::json j = parse_json_file(json_file);

    if (j.is_null()) {
        LOG("Error: No valid data found in the JSON file.");
    } else {
        LOG(j); // Log the JSON content if parsing was successful
    }
}

void
aggregation()
{
    runScriptAndMeasureTime("scratch/server.py");
    log_server_evaluation();
}

void
send_models_to_server(std::vector<Clients_Models> clients)
{
    for (auto i : clients) {
        if (i.selected) {
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
write_successful_clients()
{
    json successful_clients_json;

    for (const auto& client : selected_clients) {
        // Check if the client successfully sent their model
        Ptr<Ipv4> ipv4 = client.node->GetObject<Ipv4>();
        Ipv4Address client_ip = ipv4->GetAddress(1, 0).GetLocal(); // Get the client's IP address

        if (endOfStreamTimes.find(client_ip) != endOfStreamTimes.end()) {
            // If the client has finished sending, log the model in JSON
            std::stringstream model_filename;
            model_filename << "models/" << client.node << ".keras";
            successful_clients_json["successful_clients"].push_back(model_filename.str());
        }
    }

    // Save the successful clients' models to a JSON file
    std::ofstream out("successful_clients.json");
    out << std::setw(4) << successful_clients_json << std::endl;
    out.close();
}

void
manager()
{
    static Time round_start;

    if (Simulator::Now() - round_start > timeout) {
        round_finished = true;
        LOG("Round timed out, not all clients were able to send " << endOfStreamTimes.size() << "/"
            << n_participating_clients);
    }

    nodes_ips = node_to_ips();
    if (round_finished) {
        if (round_number != 0) {
            LOG("Round finished at " << Simulator::Now().GetSeconds()
                << ", all clients were able to send! "
                << endOfStreamTimes.size() << "/" << n_participating_clients);
            write_successful_clients();
            aggregation();
        }

        round_cleanup();
        round_start = Simulator::Now();
        round_number++;
        round_finished = false;

        LOG("Starting round " << round_number << " at " << Simulator::Now().GetSeconds()
            << " seconds.");

        clients_info = train_clients();
        selected_clients = client_selection(n_participating_clients, clients_info);

        send_models_to_server(selected_clients);
    }

    round_finished = finished_transmission(nodes_ips, selected_clients);

    Simulator::Schedule(Seconds(1), &manager);
}

// void
// read_json()
// {
//     // Helper function to parse JSON file
//     auto parse_json_file = [](const std::string& filepath) -> json {
//         std::ifstream ifs(filepath);
//         json j;
//         ifs >> j;
//         return j;
//     };

//     std::string json_file = "models/0x559d3f606780_model_sizes.json";

//     json j = parse_json_file(json_file);
//     auto l = j["layer_importances"];

//     for (auto i : l) {
//         LOG(i);
//         i.at(0);
//         int index = i.at(0);
//         std::string layer_name = i.at(1);
//         double acc_drop = i.at(2);
//     }
// }

// Main function
int
main(int argc, char* argv[])
{
    // read_json();
    // getchar();

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
    Config::SetDefault("ns3::ComponentCarrier::UlBandwidth", UintegerValue(15));
    Config::SetDefault("ns3::ComponentCarrier::DlBandwidth", UintegerValue(15));

    Config::SetDefault("ns3::ComponentCarrier::PrimaryCarrier", BooleanValue(true));
    Config::SetDefault("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteSpectrumPhy::DataErrorModelEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));
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

    // Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(43.0));
    // lower the ue tx power for more challenging transmission
    Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(5.0));
    enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
    ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    mmwaveHelper->AddX2Interface(enbNodes);
    mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);
    // mmwaveHelper->EnableTraces();

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
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
    for (const auto& entry : endOfStreamTimes) {
        std::cout << "IP Address: " << entry.first
                  << " received the end signal at time: " << entry.second << " seconds."
                  << std::endl;
    }

    Simulator::Destroy();
    return 0;
}
