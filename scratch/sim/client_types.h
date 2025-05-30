#pragma once

#include "ns3/command-line.h"
#include "ns3/config-store-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/isotropic-antenna-model.h"
#include "ns3/lte-helper.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/point-to-point-helper.h"

using namespace ns3;

struct ClientModels {
    Ptr<Node> node;
    int nodeTrainingTime;
    int nodeModelSize;
    bool selected;
    double rsrp;
    double sinr;
    double accuracy;

    // Constructor with all parameters
    ClientModels(Ptr<Node> n, int t, int b, bool s, double r, double sin, double acc);

    // Constructor without selected, rsrp, and sinr
    ClientModels(Ptr<Node> n, int t, int b, double r, double sin, double acc);

    // Explicitly delete the default constructor
    // Clients_Models() = delete;
};

// Overload the '<<' operator for Clients_Models to display RSRP and SINR
std::ostream& operator<<(std::ostream& os, const ClientModels& model);

struct NodesIps {
    uint32_t nodeId;
    uint32_t index;
    Ipv4Address ip;

    NodesIps(int n, int i, Ipv4Address ia);
};
