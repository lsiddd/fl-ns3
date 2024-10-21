#include "client_types.h"

// Implementation of Clients_Models constructors
Clients_Models::Clients_Models(Ptr<Node> n, int t, int b, bool s, double r, double sin)
    : node(n),
      training_time(t),
      node_to_bytes(b),
      selected(s),
      rsrp(r),
      sinr(sin)
{
}

Clients_Models::Clients_Models(Ptr<Node> n, int t, int b, double r, double sin)
    : node(n),
      training_time(t),
      node_to_bytes(b),
      selected(false),
      rsrp(r),
      sinr(sin)
{
}

// Overload the '<<' operator for Clients_Models
std::ostream& operator<<(std::ostream& os, const Clients_Models& model)
{
    os << "Clients_Models { id: " << model.node->GetId()
       << ", training_time: " << model.training_time << ", node_to_bytes: " << model.node_to_bytes
       << ", selected: " << (model.selected ? "true" : "false") << ", RSRP: " << model.rsrp
       << " dBm, SINR: " << model.sinr << " dB }";
    return os;
}

// Implementation of NodesIps constructor
NodesIps::NodesIps(int n, int i, Ipv4Address ia)
    : node_id(n),
      index(i),
      ip(ia)
{
}
