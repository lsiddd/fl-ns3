#include "client_types.h"

// Implementation of Clients_Models constructors
ClientModels::ClientModels(Ptr<Node> n, int t, int b, bool s, double r, double sin, double acc)
    : node(n),
      nodeTrainingTime(t),
      nodeModelSize(b),
      selected(s),
      rsrp(r),
      sinr(sin),
      accuracy(acc)
{
}

ClientModels::ClientModels(Ptr<Node> n, int t, int b, double r, double sin, double acc)
    : node(n),
      nodeTrainingTime(t),
      nodeModelSize(b),
      selected(false),
      rsrp(r),
      sinr(sin),
      accuracy(acc)
{
}

// Overload the '<<' operator for Clients_Models
std::ostream& operator<<(std::ostream& os, const ClientModels& model)
{
    os << "Clients_Models { id: " << model.node->GetId()
       << ", training_time: " << model.nodeTrainingTime << ", node_to_bytes: " << model.nodeModelSize
       << ", selected: " << (model.selected ? "true" : "false") << ", RSRP: " << model.rsrp
       << " dBm, SINR: " << model.sinr << " dB }";
    return os;
}

// Implementation of NodesIps constructor
NodesIps::NodesIps(int n, int i, Ipv4Address ia)
    : nodeId(n),
      index(i),
      ip(ia)
{
}
