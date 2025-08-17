#pragma once

#include "client_types.h"
#include "MyApp.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include <map>
#include <vector>

using namespace ns3;

extern std::map<Ipv4Address, double> endOfStreamTimes;

class NetworkUtils {
public:
    static std::pair<uint16_t, uint16_t> getUeRntiCellid(Ptr<ns3::NetDevice> ueNetDevice);
    static std::vector<NodesIps> nodeToIps();
    static void sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size);
    static void sinkRxCallback(Ptr<const Packet> packet, const Address& from);
    static bool checkFinishedTransmission(const std::vector<NodesIps>& all_nodes_ips,
                                         const std::vector<ClientModels>& selected_clients_for_round);
    static void roundCleanup();

private:
    static const uint32_t writeSize = 1448;
};

extern uint8_t data[];
extern uint8_t dataFin[];