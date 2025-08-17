#include "utils.h"
#include "ns3/log.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstdlib>

NS_LOG_COMPONENT_DEFINE("Utils");

using namespace ns3;

void ReportUePhyMetricsFromTrace(unsigned long arg1, unsigned short arg2, unsigned short arg3)
{
    NS_LOG_INFO("ReportUePhyMetricsFromTrace (3 args) invoked. Arg1: " << arg1 << ", Arg2: " << (unsigned int)arg2 << ", Arg3: " << (unsigned int)arg3
                                                                       << ". The interpretation of these arguments is UNCERTAIN for RSRP/SINR. "
                                                                       << "Actual metrics depend on the specific trace source signature in your ns-3 LTE version. "
                                                                       << "This callback may NOT provide RSRP/SINR directly and current implementation does NOT populate sinrUe/rsrpUe maps. "
                                                                       << "Investigate fl-ns3/src/lte/model/lte-ue-phy.cc for 'ReportCurrentCellRsrpSinr' trace source details.");
}

std::streamsize getFileSize(const std::string &filename)
{
    NS_LOG_INFO("getFileSize: Attempting to get size of file: '" << filename << "'");

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        NS_LOG_ERROR("getFileSize: Could not open file: '" << filename << "'. Returning size 0.");
        return 0;
    }
    std::streamsize size = file.tellg();
    NS_LOG_INFO("getFileSize: File '" << filename << "' size: " << size << " bytes.");
    file.close();
    return size;
}

std::string extractModelPath(const std::string &input)
{
    NS_LOG_INFO("extractModelPath: Extracting model path from input string: '" << input << "'");

    const std::string modelFlag = "--model ";
    const std::string extension = ".keras";
    const std::string modelSuffix = "_model";

    size_t modelPos = input.find(modelFlag);
    NS_LOG_INFO("extractModelPath: Searching for '" << modelFlag << "'. Found at pos: " << modelPos);
    if (modelPos == std::string::npos)
    {
        NS_LOG_INFO("extractModelPath: '" << modelFlag << "' not found. Returning empty string.");
        return "";
    }

    size_t start = modelPos + modelFlag.length();
    size_t end = input.find(" ", start);
    if (end == std::string::npos)
    {
        NS_LOG_INFO("extractModelPath: No space found after model path, using end of string.");
        end = input.length();
    }
    NS_LOG_INFO("extractModelPath: Model path candidate start: " << start << ", end: " << end);

    std::string modelPath = input.substr(start, end - start);
    NS_LOG_INFO("extractModelPath: Initial extracted model path: '" << modelPath << "'");

    size_t extensionPos = modelPath.find(extension);
    NS_LOG_INFO("extractModelPath: Searching for extension '" << extension << "'. Found at pos: " << extensionPos);
    if (extensionPos != std::string::npos)
    {
        modelPath = modelPath.substr(0, extensionPos);
        NS_LOG_INFO("extractModelPath: Removed extension. Path now: '" << modelPath << "'");
    }

    size_t modelSuffixPos = modelPath.rfind(modelSuffix);
    NS_LOG_INFO("extractModelPath: Searching for suffix '" << modelSuffix << "'. Found at pos: " << modelSuffixPos);
    if (modelSuffixPos != std::string::npos)
    {
        modelPath = modelPath.substr(0, modelSuffixPos);
        NS_LOG_INFO("extractModelPath: Removed suffix. Path now: '" << modelPath << "'");
    }

    NS_LOG_INFO("extractModelPath: Final extracted model path: '" << modelPath << "' from input '" << input << "'");
    return modelPath;
}