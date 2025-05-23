#include "MyApp.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include <iostream> // For std::endl

NS_LOG_COMPONENT_DEFINE("MyApp");

MyApp::MyApp() = default;

MyApp::~MyApp() {
    NS_LOG_DEBUG("MyApp destructor called for socket " << m_socket);
    m_socket = nullptr;
}

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
                  uint32_t nPackets, DataRate dataRate, uint32_t writeSize, uint8_t* data, uint8_t* dataFin) {
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_startTime = Simulator::Now();
    m_data = data;
    m_data_fin = dataFin;
    m_writeSize = writeSize;
    NS_LOG_INFO("MyApp Setup: Socket=" << socket << ", Peer=" << address
                                       << ", PacketSize=" << packetSize << ", NumPackets=" << nPackets
                                       << ", DataRate=" << dataRate << ", WriteSize=" << writeSize);
}

void MyApp::StartApplication() {
    m_running = true;
    m_packetsSent = 0;
    NS_LOG_INFO("MyApp StartApplication: Starting at " << Simulator::Now().GetSeconds() << "s.");

    if (!m_socket) {
        NS_FATAL_ERROR("Socket not initialized");
    }

    m_socket->Bind();
    m_socket->Connect(m_peer);
    NS_LOG_DEBUG("MyApp StartApplication: Socket bound and connected to " << m_peer);
    SendPacket();
}

void MyApp::StopApplication() {
    m_running = false;
    NS_LOG_INFO("MyApp StopApplication: Stopping at " << Simulator::Now().GetSeconds() << "s. Packets sent: " << m_packetsSent);

    if (m_sendEvent.IsPending()) {
        Simulator::Cancel(m_sendEvent);
        NS_LOG_DEBUG("MyApp StopApplication: Cancelled pending send event.");
    }

    if (m_socket) {
        m_socket->Close();
        NS_LOG_DEBUG("MyApp StopApplication: Socket closed.");
    }
}

void MyApp::SendPacket() {
    if (!m_running) {
        NS_LOG_DEBUG("MyApp SendPacket: Application not running, skipping packet send.");
        return;
    }

    Ptr<Packet> packet;
    if (m_packetsSent + 1 == m_nPackets) {
        packet = Create<Packet>(m_data_fin, m_writeSize);
        NS_LOG_INFO("MyApp SendPacket: Sending final packet (num " << m_packetsSent + 1 << "/" << m_nPackets << ") of size " << m_writeSize << " (FIN signal).");
    } else {
        packet = Create<Packet>(m_data, m_writeSize);
        NS_LOG_DEBUG("MyApp SendPacket: Sending data packet (num " << m_packetsSent + 1 << "/" << m_nPackets << ") of size " << m_writeSize << ".");
    }

    int bytesSent = m_socket->Send(packet);
    if (bytesSent < 0) {
        NS_LOG_ERROR("MyApp SendPacket: Error sending packet. Bytes sent: " << bytesSent);
    } else if (bytesSent != (int)m_writeSize) {
        NS_LOG_WARN("MyApp SendPacket: Sent " << bytesSent << " bytes, expected " << m_writeSize << " bytes.");
    }

    ++m_packetsSent;

    if (m_packetsSent < m_nPackets) {
        ScheduleTx();
    } else {
        NS_LOG_INFO("MyApp SendPacket: All " << m_nPackets << " packets sent for this stream.");
    }
}

void MyApp::ScheduleTx() {
    if (m_running) {
        double seconds = static_cast<double>(m_packetSize * 8) / m_dataRate.GetBitRate();
        Time tNext = Seconds(seconds);
        m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
        NS_LOG_DEBUG("MyApp ScheduleTx: Next packet scheduled for " << tNext.GetSeconds() << "s from now (at " << (Simulator::Now() + tNext).GetSeconds() << "s).");
    } else {
        NS_LOG_DEBUG("MyApp ScheduleTx: Application not running, not scheduling next transmission.");
    }
}