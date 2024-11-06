#include "MyApp.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/packet.h"

NS_LOG_COMPONENT_DEFINE("MyApp");


// static const uint32_t writeSize = 2500;
// uint8_t data[writeSize] = {'g'};
// uint8_t dataFin[writeSize] = {'b'};

MyApp::MyApp() = default;

MyApp::~MyApp()
{
    m_socket = nullptr;
}

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
                  uint32_t nPackets, DataRate dataRate, uint32_t writeSize, uint8_t* data, uint8_t* dataFin)
{
    m_socket = socket;
    m_peer = address;
    m_packetSize = packetSize;
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_startTime = Simulator::Now();
    m_data = data;
    m_data_fin = dataFin;
    m_writeSize = writeSize;
}

void MyApp::StartApplication()
{
    m_running = true;
    m_packetsSent = 0;

    if (!m_socket) {
        NS_FATAL_ERROR("Socket not initialized");
    }

    m_socket->Bind();
    m_socket->Connect(m_peer);
    SendPacket();
}

void MyApp::StopApplication()
{
    m_running = false;

    if (m_sendEvent.IsPending()) {
        Simulator::Cancel(m_sendEvent);
    }

    if (m_socket) {
        m_socket->Close();
    }
}

void MyApp::SendPacket()
{
    Ptr<Packet> packet = Create<Packet>(m_packetSize);

    if (m_packetsSent + 1 == m_nPackets) {
        m_socket->Send(m_data_fin, m_writeSize, 0);
    } else {
        m_socket->Send(m_data, m_writeSize, 0);
    }

    ++m_packetsSent;

    if (m_packetsSent < m_nPackets) {
        ScheduleTx();
    }
}

void MyApp::ScheduleTx()
{
    if (m_running) {
        double seconds = static_cast<double>(m_packetSize * 8) / m_dataRate.GetBitRate();
        Time tNext = Seconds(seconds);
        m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
    }
}

