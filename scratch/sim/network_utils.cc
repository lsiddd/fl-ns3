#include "network_utils.h"
#include "network_setup.h"
#include "ns3/log.h"
#include <chrono>
#include <iomanip>
#include <algorithm>

NS_LOG_COMPONENT_DEFINE("NetworkUtils");

std::map<Ipv4Address, double> endOfStreamTimes;

static const uint32_t writeSize = 1448;
uint8_t data[writeSize];
uint8_t dataFin[writeSize];

struct DataInitializer {
    DataInitializer() {
        NS_LOG_INFO("DataInitializer: Initializing global data arrays 'data' and 'dataFin'.");
        std::fill_n(data, writeSize, 'g');
        std::fill_n(dataFin, writeSize, 'b');
        NS_LOG_INFO("DataInitializer: Global data arrays 'data' and 'dataFin' initialized successfully.");
    }
};
DataInitializer globalDataInitializer;

std::pair<uint16_t, uint16_t> NetworkUtils::getUeRntiCellid(Ptr<ns3::NetDevice> ueNetDevice) {
    NS_LOG_DEBUG("getUeRntiCellid: Attempting to get RNTI and CellID for NetDevice: " << (ueNetDevice ? ueNetDevice->GetInstanceTypeId().GetName() : "null"));

    if (!ueNetDevice) {
        NS_LOG_DEBUG("getUeRntiCellid: Input ueNetDevice is null. Returning {0, 0}.");
        return {0, 0};
    }

    auto lteDevice = ueNetDevice->GetObject<LteUeNetDevice>();
    if (!lteDevice) {
        NS_LOG_DEBUG("getUeRntiCellid: NetDevice is not an LteUeNetDevice or GetObject failed. Device Type: " << ueNetDevice->GetInstanceTypeId().GetName() << ". Returning {0, 0}.");
        return {0, 0};
    }

    NS_LOG_DEBUG("getUeRntiCellid: LteUeNetDevice found: " << lteDevice);

    if (!lteDevice->GetRrc()) {
        NS_LOG_DEBUG("getUeRntiCellid: LteUeNetDevice " << lteDevice << " has no RRC instance. Returning {0, 0}.");
        return {0, 0};
    }

    LteUeRrc::State rrcState = lteDevice->GetRrc()->GetState();
    NS_LOG_DEBUG("getUeRntiCellid: LteUeRrc State: " << rrcState);

    if (rrcState != LteUeRrc::CONNECTED_NORMALLY && rrcState != LteUeRrc::CONNECTED_HANDOVER) {
        NS_LOG_DEBUG("getUeRntiCellid: UE RRC not in a connected state (Current state: " << rrcState << "). Returning {0, 0}.");
        return {0, 0};
    }

    auto rnti = lteDevice->GetRrc()->GetRnti();
    auto cellid = lteDevice->GetRrc()->GetCellId();
    NS_LOG_DEBUG("getUeRntiCellid: Successfully retrieved RNTI: " << rnti << ", CellID: " << cellid << " for UE connected to eNB.");
    return std::make_pair(rnti, cellid);
}

std::vector<NodesIps> NetworkUtils::nodeToIps() {
    NS_LOG_INFO("nodeToIps: Starting to map UE Node IDs to their IP addresses.");

    std::vector<NodesIps> nodes_ips_list;
    NS_LOG_DEBUG("nodeToIps: Iterating over " << ueNodes.GetN() << " UE nodes.");

    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        NS_LOG_DEBUG("nodeToIps: Processing UE Node " << i << " (Node ID: " << (ueNode ? ueNode->GetId() : -1) << ")");

        if (!ueNode) {
            NS_LOG_WARN("nodeToIps: UE Node at index " << i << " is null. Skipping.");
            continue;
        }

        Ptr<Ipv4> ipv4 = ueNode->GetObject<Ipv4>();
        if (ipv4) {
            NS_LOG_DEBUG("nodeToIps: Node " << ueNode->GetId() << " has Ipv4 object. Number of interfaces: " << ipv4->GetNInterfaces());
            if (ipv4->GetNInterfaces() > 1) {
                Ipv4InterfaceAddress iaddr = ipv4->GetAddress(1, 0);
                Ipv4Address ipAddr = iaddr.GetLocal();
                NS_LOG_DEBUG("nodeToIps: Node ID: " << ueNode->GetId() << ", UE Index: " << i << ", IP Address: " << ipAddr);
                nodes_ips_list.push_back(NodesIps(ueNode->GetId(), i, ipAddr));
            } else {
                NS_LOG_WARN("nodeToIps: Node " << ueNode->GetId() << " (UE Index " << i << ") has Ipv4 object but less than 2 interfaces (found "
                                               << ipv4->GetNInterfaces() << "). Cannot get IP from interface 1.");
            }
        } else {
            NS_LOG_WARN("nodeToIps: Node " << ueNode->GetId() << " (UE Index " << i << ") does not have an Ipv4 object. Cannot get IP.");
        }
    }
    NS_LOG_INFO("nodeToIps: Finished mapping. Found " << nodes_ips_list.size() << " UE nodes with IP addresses on interface 1.");
    return nodes_ips_list;
}

void NetworkUtils::sinkRxCallback(Ptr<const Packet> packet, const Address &from) {
    if (!packet) {
        return;
    }
    uint32_t packetSize = packet->GetSize();

    if (packetSize == 0) {
        return;
    }

    uint32_t bytes_to_check = std::min((uint32_t)100, packetSize);
    std::vector<uint8_t> buffer(bytes_to_check);
    packet->CopyData(buffer.data(), bytes_to_check);

    InetSocketAddress address = InetSocketAddress::ConvertFrom(from);
    Ipv4Address senderIp = address.GetIpv4();

    bool fin_packet = false;
    FinHeader finHeader;
    if (packet->PeekHeader(finHeader) && finHeader.IsFin()) {
        fin_packet = true;
        NS_LOG_DEBUG("sinkRxCallback: FIN header found in packet");
    }

    if (fin_packet) {
        double receiveTime = Simulator::Now().GetSeconds();
        NS_LOG_INFO(std::fixed << std::setprecision(6) << receiveTime << "s: Stream ending signal ('b') received from " << senderIp);
        if (endOfStreamTimes.find(senderIp) == endOfStreamTimes.end()) {
            endOfStreamTimes[senderIp] = receiveTime;
            NS_LOG_INFO("sinkRxCallback: Recorded end-of-stream time " << receiveTime << "s for sender " << senderIp);
        } else {
            NS_LOG_DEBUG("sinkRxCallback: Received duplicate FIN signal from " << senderIp
                                                                               << " . Original time: " << endOfStreamTimes[senderIp] << "s, new time: " << receiveTime << "s. Not updating.");
        }
    } else {
        NS_LOG_DEBUG("sinkRxCallback: Packet from " << senderIp << " is a data packet (no 'b' found in first " << bytes_to_check << " bytes).");
    }
}

void NetworkUtils::sendStream(Ptr<Node> sendingNode, Ptr<Node> receivingNode, int size) {
    std::vector<uint8_t> data(writeSize, 'g');
    std::vector<uint8_t> dataFin(writeSize, 'b');
    NS_LOG_INFO("sendStream: Called with sendingNode=" << (sendingNode ? sendingNode->GetId() : 0)
                                                       << ", receivingNode=" << (receivingNode ? receivingNode->GetId() : 0)
                                                       << ", size=" << size << " bytes.");

    if (!sendingNode || !receivingNode) {
        NS_LOG_ERROR("sendStream: One or both nodes are null. Sending Node: " << sendingNode << ", Receiving Node: " << receivingNode << ". Aborting stream setup.");
        return;
    }
    NS_LOG_INFO("sendStream: Initiating stream from Node " << sendingNode->GetId() << " to Node " << receivingNode->GetId() << " for " << size << " bytes.");

    static uint16_t port_counter = 5000;
    uint16_t current_port = port_counter++;
    NS_LOG_DEBUG("sendStream: Using port " << current_port << " for this stream.");

    if (size == 0) {
        NS_LOG_INFO("sendStream: Requested size is 0. Nothing to send from Node " << sendingNode->GetId() << " to Node " << receivingNode->GetId() << ". Skipping stream setup.");
        return;
    }

    uint32_t nPackets = (size + writeSize - 1) / writeSize;
    if (nPackets == 0 && size > 0) {
        nPackets = 1;
        NS_LOG_DEBUG("sendStream: Size " << size << " is less than writeSize " << writeSize << ", adjusting nPackets to 1.");
    } else if (nPackets > 0 && size % writeSize == 0) {
        nPackets += 1;
        NS_LOG_DEBUG("sendStream: Size " << size << " is a multiple of writeSize " << writeSize << ", adjusting nPackets to " << nPackets << " for FIN signal.");
    } else {
        NS_LOG_DEBUG("sendStream: nPackets calculated: " << nPackets);
    }

    Ptr<Ipv4> ipv4Receiver = receivingNode->GetObject<Ipv4>();
    if (!ipv4Receiver) {
        NS_LOG_ERROR("sendStream: No Ipv4 object found on receiving node " << receivingNode->GetId() << ". Cannot determine destination IP. Aborting.");
        return;
    }

    Ipv4Address ipAddrReceiver;
    if (receivingNode->GetId() == remoteHostContainer.Get(0)->GetId()) {
        ipAddrReceiver = ipv4Receiver->GetAddress(1, 0).GetLocal();
    } else {
        ipAddrReceiver = ipv4Receiver->GetAddress(1, 0).GetLocal();
    }

    if (ipAddrReceiver == Ipv4Address()) {
        NS_LOG_ERROR("sendStream: Failed to get valid IP address for receiving node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    NS_LOG_DEBUG("sendStream: Receiving Node " << receivingNode->GetId() << " IP Address: " << ipAddrReceiver);

    NS_LOG_INFO(Simulator::Now().GetSeconds()
                << "s: Node " << sendingNode->GetId() << " starting stream to "
                << receivingNode->GetId() << " (" << ipAddrReceiver << ":" << current_port << "), " << size << " bytes, "
                << nPackets << " packets, each up to " << writeSize << " payload bytes.");

    Address sinkAddress(InetSocketAddress(ipAddrReceiver, current_port));
    PacketSinkHelper packetSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), current_port));
    NS_LOG_DEBUG("sendStream: Installing TCP PacketSink on Node " << receivingNode->GetId() << " at " << InetSocketAddress(Ipv4Address::GetAny(), current_port));
    ApplicationContainer sinkApps = packetSinkHelper.Install(receivingNode);

    Time streamStartTime = MilliSeconds(200);
    sinkApps.Start(streamStartTime);

    NS_LOG_INFO("sendStream: PacketSink application scheduled to start at " << streamStartTime.GetSeconds() << "s on Node " << receivingNode->GetId() << ".");

    if (sinkApps.GetN() == 0) {
        NS_LOG_ERROR("sendStream: Failed to install PacketSink on Node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApps.Get(0));
    if (!sink) {
        NS_LOG_ERROR("sendStream: Failed to cast Application to PacketSink on Node " << receivingNode->GetId() << ". Aborting.");
        return;
    }
    sink->TraceConnectWithoutContext("Rx", MakeCallback(&NetworkUtils::sinkRxCallback));
    NS_LOG_DEBUG("sendStream: Connected Rx trace of PacketSink on Node " << receivingNode->GetId() << " to sinkRxCallback.");

    NS_LOG_DEBUG("sendStream: Creating TCP socket on sending Node " << sendingNode->GetId());
    Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(sendingNode, TcpSocketFactory::GetTypeId());
    if (!ns3TcpSocket) {
        NS_LOG_ERROR("sendStream: Failed to create TCP socket on sending Node " << sendingNode->GetId() << ". Aborting.");
        return;
    }
    NS_LOG_DEBUG("sendStream: TCP socket created successfully on Node " << sendingNode->GetId());

    NS_LOG_DEBUG("sendStream: Creating MyApp application object.");
    Ptr<MyApp> app = CreateObject<MyApp>();
    NS_LOG_INFO("sendStream: Setting up MyApp on Node " << sendingNode->GetId() << " to send to " << sinkAddress
                                                        << ", packetSizeForScheduling=" << writeSize << ", nPackets=" << nPackets
                                                        << ", dataRate=1Mbps, actualSendSize=" << writeSize << " bytes.");
    app->Setup(ns3TcpSocket, sinkAddress, writeSize, nPackets,
               DataRate("1Mbps"), writeSize, data, dataFin);

    sendingNode->AddApplication(app);
    NS_LOG_DEBUG("sendStream: MyApp added to Node " << sendingNode->GetId());

    app->SetStartTime(streamStartTime);

    NS_LOG_INFO("sendStream: MyApp scheduled to start at " << streamStartTime.GetSeconds() << "s on Node " << sendingNode->GetId());
}

bool NetworkUtils::checkFinishedTransmission(const std::vector<NodesIps> &all_nodes_ips,
                               const std::vector<ClientModels> &selected_clients_for_round) {
    NS_LOG_DEBUG("checkFinishedTransmission: Checking finished transmissions. Number of all_nodes_ips: " << all_nodes_ips.size()
                                                                                                         << ", Number of selected_clients_for_round: " << selected_clients_for_round.size());

    if (selected_clients_for_round.empty()) {
        NS_LOG_INFO("checkFinishedTransmission: No clients selected for this round. Transmission considered finished.");
        return true;
    }

    size_t finished_count = 0;
    NS_LOG_DEBUG("checkFinishedTransmission: Iterating through " << selected_clients_for_round.size() << " selected clients to check transmission status.");
    for (const auto &client_model : selected_clients_for_round) {
        if (!client_model.node) {
            NS_LOG_WARN("checkFinishedTransmission: Client model has a null Ptr<Node>. Skipping this client.");
            continue;
        }
        uint32_t clientNodeId = client_model.node->GetId();
        NS_LOG_DEBUG("checkFinishedTransmission: Checking client with Node ID: " << clientNodeId);

        Ipv4Address client_ip;
        bool ip_found = false;
        for (const auto &nip : all_nodes_ips) {
            if (nip.nodeId == clientNodeId) {
                client_ip = nip.ip;
                ip_found = true;
                NS_LOG_DEBUG("checkFinishedTransmission: Found IP " << client_ip << " for Node ID " << clientNodeId);
                break;
            }
        }

        if (ip_found) {
            if (endOfStreamTimes.count(client_ip)) {
                NS_LOG_DEBUG("checkFinishedTransmission: Client " << clientNodeId << " (IP: " << client_ip << ") has finished transmission (EOS time found: " << endOfStreamTimes.at(client_ip) << "s).");
                finished_count++;
            } else {
                NS_LOG_DEBUG("checkFinishedTransmission: Client " << clientNodeId << " (IP: " << client_ip << ") has NOT finished transmission yet (no EOS time found).");
            }
        } else {
            NS_LOG_WARN("checkFinishedTransmission: Could not find IP address for client Node ID " << clientNodeId << " in all_nodes_ips. Cannot check its transmission status.");
        }
    }

    bool all_finished = (finished_count == selected_clients_for_round.size());
    NS_LOG_INFO("checkFinishedTransmission: Finished transmission check: " << finished_count << " out of " << selected_clients_for_round.size()
                                                                           << " selected clients have completed. All finished: " << (all_finished ? "YES" : "NO"));
    return all_finished;
}

void NetworkUtils::roundCleanup() {
    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: Starting round cleanup: Clearing endOfStreamTimes and stopping applications.");

    NS_LOG_INFO("roundCleanup: Clearing endOfStreamTimes map. Current size: " << endOfStreamTimes.size());
    endOfStreamTimes.clear();
    NS_LOG_INFO("roundCleanup: endOfStreamTimes map cleared.");

    NS_LOG_INFO("roundCleanup: Iterating through " << ueNodes.GetN() << " UE nodes to stop MyApp applications.");
    for (uint32_t i = 0; i < ueNodes.GetN(); i++) {
        Ptr<Node> ueNode = ueNodes.Get(i);
        if (!ueNode) {
            NS_LOG_WARN("roundCleanup: UE Node at index " << i << " is null. Skipping app cleanup for this node.");
            continue;
        }
        NS_LOG_DEBUG("roundCleanup: Node " << ueNode->GetId() << " (index " << i << "). Has " << ueNode->GetNApplications() << " applications.");

        for (int j = ueNode->GetNApplications() - 1; j >= 0; --j) {
            Ptr<Application> app = ueNode->GetApplication(j);
            if (!app) {
                NS_LOG_WARN("roundCleanup: Node " << ueNode->GetId() << ", App index " << j << " is null. Skipping.");
                continue;
            }
            NS_LOG_DEBUG("roundCleanup: Node " << ueNode->GetId() << ", checking App " << j << " (Type: " << app->GetInstanceTypeId().GetName() << ")");
            if (DynamicCast<MyApp>(app)) {
                NS_LOG_INFO("roundCleanup: Found MyApp on Node " << ueNode->GetId() << ". Setting StopTime to " << Simulator::Now().GetSeconds() << "s.");
                app->SetStopTime(Simulator::Now());
            }
        }
    }
    NS_LOG_INFO("roundCleanup: Finished processing MyApp applications on UE nodes.");

    if (remoteHostContainer.GetN() > 0) {
        Ptr<Node> serverNode = remoteHostContainer.Get(0);
        if (!serverNode) {
            NS_LOG_WARN("roundCleanup: Server node (remoteHostContainer.Get(0)) is null. Skipping PacketSink cleanup.");
        } else {
            NS_LOG_INFO("roundCleanup: Processing server Node " << serverNode->GetId() << " to stop PacketSink applications. Has " << serverNode->GetNApplications() << " applications.");

            for (int j = serverNode->GetNApplications() - 1; j >= 0; --j) {
                Ptr<Application> app = serverNode->GetApplication(j);
                if (!app) {
                    NS_LOG_WARN("roundCleanup: Server Node " << serverNode->GetId() << ", App index " << j << " is null. Skipping.");
                    continue;
                }
                NS_LOG_DEBUG("roundCleanup: Server Node " << serverNode->GetId() << ", checking App " << j << " (Type: " << app->GetInstanceTypeId().GetName() << ")");
                if (DynamicCast<PacketSink>(app)) {
                    NS_LOG_INFO("roundCleanup: Found PacketSink on server Node " << serverNode->GetId() << ". Setting StopTime to " << Simulator::Now().GetSeconds() << "s.");
                    app->SetStopTime(Simulator::Now());
                }
            }
            NS_LOG_INFO("roundCleanup: Finished processing PacketSink applications on server node.");
        }
    } else {
        NS_LOG_INFO("roundCleanup: No remote hosts in remoteHostContainer. Skipping PacketSink cleanup.");
    }

    NS_LOG_INFO(Simulator::Now().GetSeconds() << "s: ns-3 round cleanup finished.");
}