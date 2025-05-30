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

void
NotifyConnectionEstablishedUe(uint64_t imsi, uint16_t cellid, uint16_t rnti);

void
NotifyHandoverStartUe(std::string context, // Keep context for now, check trace if error occurs
                      uint64_t imsi,
                      uint16_t cellid,
                      uint16_t rnti,
                      uint16_t targetCellId);

void
NotifyHandoverEndOkUe(std::string context, // Keep context for now, check trace if error occurs
                       uint64_t imsi, uint16_t cellid, uint16_t rnti);

// Changed signature: removed std::string context
void
NotifyConnectionEstablishedEnb(uint64_t imsi, uint16_t cellid, uint16_t rnti);

void
NotifyHandoverStartEnb(std::string context, // Keep context for now, check trace if error occurs
                        uint64_t imsi,
                        uint16_t cellid,
                        uint16_t rnti,
                        uint16_t targetCellId);
void
NotifyHandoverEndOkEnb(std::string context, // Keep context for now, check trace if error occurs
                       uint64_t imsi, uint16_t cellid, uint16_t rnti);
