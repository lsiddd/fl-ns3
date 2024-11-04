#ifndef SIM_CLIENT_TYPES_H
#define SIM_CLIENT_TYPES_H

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

using namespace ns3;

struct Clients_Models {
    Ptr<Node> node;
    int nodeTrainingTime;
    int nodeModelSize;
    bool selected;
    double rsrp;
    double sinr;
    double accuracy;

    // Constructor with all parameters
    Clients_Models(Ptr<Node> n, int t, int b, bool s, double r, double sin, double acc);

    // Constructor without selected, rsrp, and sinr
    Clients_Models(Ptr<Node> n, int t, int b, double r, double sin, double acc);

    // Explicitly delete the default constructor
    // Clients_Models() = delete;
};

// Overload the '<<' operator for Clients_Models to display RSRP and SINR
std::ostream& operator<<(std::ostream& os, const Clients_Models& model);

struct NodesIps {
    uint32_t node_id;
    uint32_t index;
    Ipv4Address ip;

    NodesIps(int n, int i, Ipv4Address ia);
};

#endif
