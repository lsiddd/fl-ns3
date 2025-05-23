#include "client_types.h"
#include "ns3/log.h" // Include log module

NS_LOG_COMPONENT_DEFINE("ClientTypes"); // Define log component

// Implementation of Clients_Models constructors
ClientModels::ClientModels(Ptr<Node> n, int t, int b, bool s, double r, double sin, double acc)
    : node(n),
      nodeTrainingTime(t),
      nodeModelSize(b),
      selected(s),
      rsrp(r),
      sinr(sin),
      accuracy(acc) {
    NS_LOG_DEBUG("ClientModels constructor (full): NodeId=" << (node ? node->GetId() : 0)
                                                          << ", TrainingTime=" << nodeTrainingTime
                                                          << ", ModelSize=" << nodeModelSize
                                                          << ", Selected=" << (selected ? "true" : "false")
                                                          << ", RSRP=" << rsrp << ", SINR=" << sinr
                                                          << ", Accuracy=" << accuracy);
}

ClientModels::ClientModels(Ptr<Node> n, int t, int b, double r, double sin, double acc)
    : node(n),
      nodeTrainingTime(t),
      nodeModelSize(b),
      selected(false), // Default to false
      rsrp(r),
      sinr(sin),
      accuracy(acc) {
    NS_LOG_DEBUG("ClientModels constructor (partial): NodeId=" << (node ? node->GetId() : 0)
                                                             << ", TrainingTime=" << nodeTrainingTime
                                                             << ", ModelSize=" << nodeModelSize
                                                             << ", RSRP=" << rsrp << ", SINR=" << sinr
                                                             << ", Accuracy=" << accuracy);
}

// Overload the '<<' operator for Clients_Models
std::ostream& operator<<(std::ostream& os, const ClientModels& model) {
    os << "Clients_Models { id: " << (model.node ? model.node->GetId() : 0)
       << ", training_time: " << model.nodeTrainingTime << ", node_to_bytes: " << model.nodeModelSize
       << ", selected: " << (model.selected ? "true" : "false")
       << ", RSRP: " << model.rsrp << " dBm, SINR: " << model.sinr << " dB"
       << ", Accuracy: " << model.accuracy << " }";
    return os;
}

// Implementation of NodesIps constructor
NodesIps::NodesIps(int n, int i, Ipv4Address ia)
    : nodeId(n),
      index(i),
      ip(ia) {
    NS_LOG_DEBUG("NodesIps constructor: NodeId=" << nodeId << ", Index=" << index << ", IP=" << ip);
}