#include "notifications.h"

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

// Event handling functions
void NotifyConnectionEstablishedUe(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds UE IMSI " << imsi
              << ": connected to CellId " << cellid << " with RNTI " << rnti << std::endl;
}

void NotifyHandoverStartUe(std::string context,
                           uint64_t imsi,
                           uint16_t cellid,
                           uint16_t rnti,
                           uint16_t targetCellId) {
    std::cout << Simulator::Now().GetSeconds() << " seconds UE IMSI " << imsi
              << ": previously connected to CellId " << cellid << ", doing handover to CellId "
              << targetCellId << std::endl;
}

void NotifyHandoverEndOkUe(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds UE IMSI " << imsi
              << ": successful handover to CellId " << cellid << " with RNTI " << rnti << std::endl;
}

void NotifyConnectionEstablishedEnb(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds eNB CellId " << cellid
              << ": successful connection of UE with IMSI " << imsi << " RNTI " << rnti
              << std::endl;
}

void NotifyHandoverStartEnb(std::string context,
                            uint64_t imsi,
                            uint16_t cellid,
                            uint16_t rnti,
                            uint16_t targetCellId) {
    std::cout << Simulator::Now().GetSeconds() << " seconds eNB CellId " << cellid
              << ": start handover of UE with IMSI " << imsi << " RNTI " << rnti << " to CellId "
              << targetCellId << std::endl;
}

void NotifyHandoverEndOkEnb(std::string context, uint64_t imsi, uint16_t cellid, uint16_t rnti) {
    std::cout << Simulator::Now().GetSeconds() << " seconds eNB CellId " << cellid
              << ": completed handover of UE with IMSI " << imsi << " RNTI " << rnti << std::endl;
}