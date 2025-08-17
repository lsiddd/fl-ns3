#include "notifications.h"

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

NS_LOG_COMPONENT_DEFINE("Notifications");

// Event handling functions
// Changed signature: removed std::string context
void NotifyConnectionEstablishedUe(uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s UE IMSI " << imsi
              << ": connected to CellId " << cellid << " with RNTI " << rnti);
}

void NotifyHandoverStartUe(std::string context,
                           uint64_t imsi,
                           uint16_t cellid,
                           uint16_t rnti,
                           uint16_t targetCellId) {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s UE IMSI " << imsi
              << ": previously connected to CellId " << cellid << ", doing handover to CellId "
              << targetCellId);
}

void NotifyHandoverEndOkUe(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s UE IMSI " << imsi
              << ": successful handover to CellId " << cellid << " with RNTI " << rnti);
}

// Changed signature: removed std::string context
void NotifyConnectionEstablishedEnb(uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s eNB CellId " << cellid
              << ": successful connection of UE with IMSI " << imsi << " RNTI " << rnti);
}

void NotifyHandoverStartEnb(std::string context,
                            uint64_t imsi,
                            uint16_t cellid,
                            uint16_t rnti,
                            uint16_t targetCellId) {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s eNB CellId " << cellid
              << ": start handover of UE with IMSI " << imsi << " RNTI " << rnti << " to CellId "
              << targetCellId);
}

void NotifyHandoverEndOkEnb(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s eNB CellId " << cellid
              << ": completed handover of UE with IMSI " << imsi << " RNTI " << rnti);
}