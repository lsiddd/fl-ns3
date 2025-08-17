#include "fl_coordinator.h"
#include "metrics_collector.h"
#include "network_setup.h"
#include "network_utils.h"
#include "ns3/command-line.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/log.h"
#include <cstring>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SimulationMain");

std::string algorithm = "fedavg";

int main(int argc, char *argv[]) {
  LogComponentEnable("SimulationMain", LOG_LEVEL_INFO);
  LogComponentEnable("FLCoordinator", LOG_LEVEL_INFO);
  LogComponentEnable("NetworkSetup", LOG_LEVEL_INFO);
  LogComponentEnable("MetricsCollector", LOG_LEVEL_INFO);
  LogComponentEnable("NetworkUtils", LOG_LEVEL_INFO);
  // LogComponentEnable("MyApp", LOG_LEVEL_DEBUG);
  LogComponentEnable("ClientTypes", LOG_LEVEL_INFO);
  LogComponentEnable("DataFrame", LOG_LEVEL_DEBUG);
  LogComponentEnable("Notifications", LOG_LEVEL_INFO);
  // LogComponentEnable("TcpSocket", LOG_LEVEL_DEBUG);
  // LogComponentEnable("TcpSocketBase", LOG_LEVEL_DEBUG);

  NS_LOG_INFO("Starting FL-NS3 Simulation");

  NetworkSetup::configureDefaults();
  MetricsCollector::initializeDataFrames();

  CommandLine cmd;
  cmd.AddValue("algorithm", "FL algorithm (ns-3 perspective, less relevant now)", algorithm);
  cmd.Parse(argc, argv);

  if (!FLCoordinator::initializeFlApi()) {
    NS_LOG_ERROR("Failed to initialize FL API. Exiting.");
    return 1;
  }

  Ptr<LteHelper> mmwaveHelper;
  Ptr<PointToPointEpcHelper> epcHelper;
  NetworkSetup::setupCoreNetwork(mmwaveHelper, epcHelper);
  NetworkSetup::setupNodes(numberOfEnbs, numberOfUes);
  NetworkSetup::setupMobility(useStaticClients, scenarioSize, epcHelper);
  NetworkSetup::setupDevicesAndIp(mmwaveHelper, epcHelper);
  NetworkSetup::setupRouting(epcHelper);
  MetricsCollector::setupRsrpSinrTracing();

  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();
  NS_LOG_INFO("FlowMonitor installed.");

  Simulator::Schedule(Seconds(2.0), &FLCoordinator::manager);
  Simulator::Schedule(Seconds(1.0), &MetricsCollector::networkInfo, monitor);
  NS_LOG_INFO("Manager and networkInfo functions scheduled.");

  // NetworkSetup::setupAnimation();

  Simulator::Stop(Seconds(simStopTime));
  NS_LOG_INFO("Starting ns-3 Simulation. Simulation will stop at " << simStopTime << "s.");
  Simulator::Run();
  NS_LOG_INFO("ns-3 Simulation Finished.");

  NS_LOG_INFO("Stopping Python FL API server...");
  int kill_status = system("pkill -f 'python3 scratch/sim/fl_api.py'");

  if (kill_status == -1) {
    NS_LOG_ERROR("Failed to execute pkill command: " << std::strerror(errno));
  } else {
    if (WIFEXITED(kill_status)) {
      const int exit_code = WEXITSTATUS(kill_status);
      if (exit_code == 0) {
        NS_LOG_INFO("Successfully terminated Python FL API server");
      } else if (exit_code == 1) {
        NS_LOG_WARN("Python server not found (already terminated?)");
      } else {
        NS_LOG_WARN("pkill exited abnormally (code: " << exit_code << ")");
      }
    } else {
      NS_LOG_WARN("pkill terminated by signal: " << WTERMSIG(kill_status));
    }
  }

  MetricsCollector::exportDataFrames();
  Simulator::Destroy();
  NS_LOG_INFO("Simulator Destroyed.");
  return 0;
}