#include "network_setup.h"
#include "notifications.h"
#include "ns3/log.h"

NS_LOG_COMPONENT_DEFINE("NetworkSetup");

std::map<int, ClientInfo> client_info;
std::map<int, EnbInfo> enb_info;

const double simStopTime = 400.0;
const int numberOfUes = 10;
const int numberOfEnbs = 5;
const int scenarioSize = 1000;
bool useStaticClients = true;

NodeContainer ueNodes;
NodeContainer enbNodes;
NodeContainer remoteHostContainer;
NetDeviceContainer enbDevs;
NetDeviceContainer ueDevs;
Ipv4Address remoteHostAddr;

void NetworkSetup::configureDefaults() {
  const uint32_t maxTxBufferSizeUm = 10 * 1024 * 1024 * 10;
  const uint32_t maxTxBufferSizeAm = 10 * 1024 * 1024;
  const uint32_t maxTxBufferSizeLowLat = 10 * 1024 * 1024;

  Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize",
                     UintegerValue(maxTxBufferSizeUm));
  Config::SetDefault("ns3::LteRlcAm::MaxTxBufferSize",
                     UintegerValue(maxTxBufferSizeAm));
  Config::SetDefault("ns3::LteRlcUmLowLat::MaxTxBufferSize",
                     UintegerValue(maxTxBufferSizeLowLat));
  Config::SetDefault("ns3::TcpL4Protocol::SocketType",
                     TypeIdValue(TcpCubic::GetTypeId()));
  Config::SetDefault("ns3::TcpSocketBase::MinRto",
                     TimeValue(MilliSeconds(200)));
  Config::SetDefault("ns3::Ipv4L3Protocol::FragmentExpirationTimeout",
                     TimeValue(Seconds(2)));
  Config::SetDefault("ns3::TcpSocket::SegmentSize",
                     UintegerValue(1448));
  Config::SetDefault("ns3::TcpSocket::DelAckCount", UintegerValue(1));
  uint32_t sndRcvBufSize = 131072 * 10;
  Config::SetDefault("ns3::TcpSocket::SndBufSize",
                     UintegerValue(sndRcvBufSize));
  Config::SetDefault("ns3::TcpSocket::RcvBufSize",
                     UintegerValue(sndRcvBufSize));
  Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(false));
  Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(20.0));
  NS_LOG_INFO("NS-3 default configurations applied.");
}

void NetworkSetup::setupCoreNetwork(Ptr<LteHelper> &mmwaveHelper, Ptr<PointToPointEpcHelper> &epcHelper) {
  mmwaveHelper = CreateObject<LteHelper>();
  epcHelper = CreateObject<PointToPointEpcHelper>();
  mmwaveHelper->SetEpcHelper(epcHelper);
  mmwaveHelper->SetSchedulerType("ns3::RrFfMacScheduler");
  mmwaveHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
  NS_LOG_INFO("LTE Helper and EPC Helper created and configured.");

  ConfigStore inputConfig;
  inputConfig.ConfigureDefaults();

  setupPgwRemoteHostConnection(epcHelper);
}

void NetworkSetup::setupPgwRemoteHostConnection(Ptr<PointToPointEpcHelper> epcHelper) {
  Ptr<Node> pgw = epcHelper->GetPgwNode();
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  InternetStackHelper internet;
  internet.Install(remoteHostContainer);
  NS_LOG_INFO("PGW and RemoteHost created and InternetStack installed on RemoteHost.");

  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
  p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
  p2ph.SetChannelAttribute("Delay", TimeValue(MicroSeconds(1)));
  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
  remoteHostAddr = internetIpIfaces.GetAddress(1);
  NS_LOG_INFO("Point-to-Point link between PGW and RemoteHost configured. RemoteHost IP: " << remoteHostAddr);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
      ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                             Ipv4Mask("255.0.0.0"), 1);
  NS_LOG_INFO("Static route added on RemoteHost for UE network.");
}

void NetworkSetup::setupNodes(int numberOfEnbs, int numberOfUes) {
  enbNodes.Create(numberOfEnbs);
  ueNodes.Create(numberOfUes);
  NS_LOG_INFO("Created " << numberOfEnbs << " eNBs and " << numberOfUes << " UEs.");

  for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
    enb_info[i] = EnbInfo();
  }

  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    client_info[i] = ClientInfo();
  }
}

void NetworkSetup::setupMobility(bool useStaticClients, int scenarioSize, Ptr<PointToPointEpcHelper> epcHelper) {
  MobilityHelper enbmobility;
  Ptr<RandomRectanglePositionAllocator> enbPositionAlloc =
      CreateObject<RandomRectanglePositionAllocator>();
  enbPositionAlloc->SetAttribute(
      "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                       std::to_string(scenarioSize) + "]"));
  enbPositionAlloc->SetAttribute(
      "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                       std::to_string(scenarioSize) + "]"));

  enbmobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  enbmobility.SetPositionAllocator(enbPositionAlloc);
  enbmobility.Install(enbNodes);
  NS_LOG_INFO("eNBs installed with ConstantPositionMobilityModel and random positions within scenario size.");

  for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
    Ptr<MobilityModel> mob = enbNodes.Get(i)->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();
    enb_info[i].x_pos = pos.x;
    enb_info[i].y_pos = pos.y;
    NS_LOG_INFO("eNB " << i << " position: (" << pos.x << ", " << pos.y << ")");
  }

  MobilityHelper uemobility;
  if (useStaticClients) {
    NS_LOG_INFO("Installing ConstantPositionMobilityModel for static UEs with random positions.");
    uemobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<RandomRectanglePositionAllocator> uePositionAlloc =
        CreateObject<RandomRectanglePositionAllocator>();
    uePositionAlloc->SetAttribute(
        "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                         std::to_string(scenarioSize) + "]"));
    uePositionAlloc->SetAttribute(
        "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" +
                         std::to_string(scenarioSize) + "]"));

    uemobility.SetPositionAllocator(uePositionAlloc);
    uemobility.Install(ueNodes);
    NS_LOG_INFO("Static UEs installed with ConstantPositionMobilityModel and random positions within scenario size.");
  } else {
    NS_LOG_INFO("Installing RandomWalk2dMobilityModel for mobile UEs.");
    std::string walkBounds = "0|" + std::to_string(scenarioSize) + "|0|" +
                             std::to_string(scenarioSize);
    uemobility.SetMobilityModel(
        "ns3::RandomWalk2dMobilityModel", "Mode", StringValue("Time"), "Time",
        StringValue("2s"), "Speed",
        StringValue("ns3::ConstantRandomVariable[Constant=20.0]"),
        "Bounds", StringValue(walkBounds));
    uemobility.Install(ueNodes);
    NS_LOG_INFO("Mobile UEs installed with RandomWalk2dMobilityModel within scenario size bounds.");
  }

  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    Ptr<MobilityModel> mob = ueNodes.Get(i)->GetObject<MobilityModel>();
    Vector pos = mob->GetPosition();
    client_info[i].x_pos = pos.x;
    client_info[i].y_pos = pos.y;
    NS_LOG_INFO("UE " << i << " position: (" << pos.x << ", " << pos.y << ")");
  }

  Ptr<Node> pgw = epcHelper->GetPgwNode();
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  enbmobility.Install(pgw);
  enbmobility.Install(remoteHost);
  NS_LOG_INFO("Mobility models installed on PGW and RemoteHost for NetAnim.");
}

void NetworkSetup::setupDevicesAndIp(Ptr<LteHelper> mmwaveHelper, Ptr<PointToPointEpcHelper> epcHelper) {
  enbDevs = mmwaveHelper->InstallEnbDevice(enbNodes);
  ueDevs = mmwaveHelper->InstallUeDevice(ueNodes);
  InternetStackHelper internet;
  internet.Install(ueNodes);
  epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
  mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);
  NS_LOG_INFO("eNB and UE devices installed. UE IP addresses assigned. UEs attached to closest eNB.");

// scratch/sim/network_setup.cc

for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    Ptr<NetDevice> ueDevice = ueDevs.Get(i);
    Ptr<LteUeNetDevice> lteUeDevice = ueDevice->GetObject<LteUeNetDevice>();

    // Add this check to ensure the UE is connected
    if (!lteUeDevice || !lteUeDevice->GetRrc() || 
        (lteUeDevice->GetRrc()->GetState() != LteUeRrc::CONNECTED_NORMALLY &&
         lteUeDevice->GetRrc()->GetState() != LteUeRrc::CONNECTED_HANDOVER))
    {
        NS_LOG_WARN("UE " << i << " is not in a connected state. Skipping attachment info.");
        Ptr<Ipv4> ipv4 = ueNodes.Get(i)->GetObject<Ipv4>();
        if (ipv4 && ipv4->GetNInterfaces() > 1) {
            Ipv4Address ueIp = ipv4->GetAddress(1, 0).GetLocal();
            NS_LOG_INFO("UE " << i << " IP address: " << ueIp);
        }
        continue; // Skip the rest of the loop for this unconnected UE
    }

    uint16_t enbCellId = lteUeDevice->GetRrc()->GetCellId();
    client_info[i].serving_enb = enbCellId;

    Ptr<Ipv4> ipv4 = ueNodes.Get(i)->GetObject<Ipv4>();
    Ipv4Address ueIp = ipv4->GetAddress(1, 0).GetLocal();
    NS_LOG_INFO("UE " << i << " IP address: " << ueIp);

    double dist = ueNodes.Get(i)->GetObject<MobilityModel>()->GetDistanceFrom(enbNodes.Get(enbCellId)->GetObject<MobilityModel>());
    NS_LOG_INFO("UE " << i << " is attached to eNB " << enbCellId << " with distance " << dist << "m");
}
}

void NetworkSetup::setupRouting(Ptr<PointToPointEpcHelper> epcHelper) {
  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
    Ptr<Node> ueNode = ueNodes.Get(i);
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
  }
  NS_LOG_INFO("Static routes set for UEs.");
}

void NetworkSetup::setupAnimation() {
  AnimationInterface anim("fl_api_mmwave_animation.xml");
  for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    anim.UpdateNodeDescription(ueNodes.Get(i), "UE");
    anim.UpdateNodeColor(ueNodes.Get(i), 255, 0, 0);
  }
  for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
    anim.UpdateNodeDescription(enbNodes.Get(i), "ENB");
    anim.UpdateNodeColor(enbNodes.Get(i), 0, 255, 0);
  }
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  anim.UpdateNodeDescription(remoteHost, "RH_FL_Server");
  anim.UpdateNodeColor(remoteHost, 0, 0, 255);
  NS_LOG_INFO("NetAnim configuration complete.");

  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedEnb));
  Config::ConnectWithoutContext(
      "/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
      MakeCallback(&NotifyConnectionEstablishedUe));
  NS_LOG_INFO("LTE ConnectionEstablished trace sources connected.");
}