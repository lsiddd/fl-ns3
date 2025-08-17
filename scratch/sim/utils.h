#pragma once

#include "ns3/core-module.h"
#include <string>

using namespace ns3;

int64_t runScriptAndMeasureTime(const std::string& scriptPath);
std::string extractModelPath(const std::string& input);
std::streamsize getFileSize(const std::string& filename);
void ReportUePhyMetricsFromTrace(unsigned long arg1, unsigned short arg2, unsigned short arg3);
