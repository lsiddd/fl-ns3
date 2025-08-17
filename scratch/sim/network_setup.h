#pragma once

#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"

using namespace ns3;

extern NodeContainer ueNodes;
extern NodeContainer enbNodes;
extern NodeContainer remoteHostContainer;
extern NetDeviceContainer enbDevs;
extern NetDeviceContainer ueDevs;
extern Ipv4Address remoteHostAddr;

class NetworkSetup {
public:
    static void configureDefaults();
    static void setupCoreNetwork(Ptr<LteHelper> &mmwaveHelper, Ptr<PointToPointEpcHelper> &epcHelper);
    static void setupNodes(int numberOfEnbs, int numberOfUes);
    static void setupMobility(bool useStaticClients, int scenarioSize, Ptr<PointToPointEpcHelper> epcHelper);
    static void setupDevicesAndIp(Ptr<LteHelper> mmwaveHelper, Ptr<PointToPointEpcHelper> epcHelper);
    static void setupRouting(Ptr<PointToPointEpcHelper> epcHelper);
    static void setupAnimation();

private:
    static void setupPgwRemoteHostConnection(Ptr<PointToPointEpcHelper> epcHelper);
};

extern const double simStopTime;
extern const int numberOfUes;
extern const int numberOfEnbs;
extern const int scenarioSize;
extern bool useStaticClients;