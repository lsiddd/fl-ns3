#pragma once

#include "dataframe.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include <map>

using namespace ns3;

extern DataFrame throughput_df;
extern DataFrame rsrp_sinr_df;
extern std::map<uint16_t, std::map<uint16_t, double>> sinrUe;
extern std::map<uint16_t, std::map<uint16_t, double>> rsrpUe;

class MetricsCollector {
public:
    static void initializeDataFrames();
    static std::pair<double, double> getRsrpSinr(uint32_t nodeIdx);
    static void updateAllClientsGlobalInfo();
    static void networkInfo(Ptr<FlowMonitor> monitor);
    static void exportDataFrames();
    static void setupRsrpSinrTracing();

    static void reportUeSinrRsrp(uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);
    static void reportUeSinrRsrp(std::string context, uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);

private:
};

void ReportUeSinrRsrp(uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);
void ReportUeSinrRsrp(std::string context, uint16_t cellId, uint16_t rnti, double rsrp, double sinr, uint8_t componentCarrierId);