#include "MyApp.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/packet.h"
#include "ns3/header.h"
#include <iostream> // For std::endl

NS_LOG_COMPONENT_DEFINE("MyApp");

NS_OBJECT_ENSURE_REGISTERED(FinHeader);

TypeId FinHeader::GetTypeId()
{
    static TypeId tid = TypeId("ns3::FinHeader")
                            .SetParent<Header>()
                            .AddConstructor<FinHeader>();
    return tid;
}

TypeId FinHeader::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t FinHeader::GetSerializedSize() const
{
    return sizeof(uint8_t); // 1 byte for boolean flag
}

void FinHeader::Serialize(Buffer::Iterator start) const
{
    start.WriteU8(m_isFin ? 1 : 0);
}

uint32_t FinHeader::Deserialize(Buffer::Iterator start)
{
    m_isFin = (start.ReadU8() == 1);
    return GetSerializedSize();
}

void FinHeader::Print(std::ostream &os) const
{
    os << "FIN=" << (m_isFin ? "true" : "false");
}

MyApp::MyApp() = default;

MyApp::~MyApp()
{
    NS_LOG_DEBUG("MyApp destructor called for socket " << m_socket);
    // NS-3 Ptr handles deallocation, setting to nullptr is good practice after Close()
    m_socket = nullptr;
}

void MyApp::Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
                  uint32_t nPackets, DataRate dataRate, uint32_t writeSize,
                  const std::vector<uint8_t> &data, const std::vector<uint8_t> &dataFin)
{
    m_socket = socket;
    m_peer = address;
    // m_packetSize = packetSize; // This member variable is not used in ScheduleTx or SendPacket. Keeping it for potential future use or removing if truly redundant.
    m_nPackets = nPackets;
    m_dataRate = dataRate;
    m_startTime = Simulator::Now();
    m_data = data;
    m_data_fin = dataFin;
    m_writeSize = writeSize;
    if (m_data.size() < m_writeSize || m_data_fin.size() < m_writeSize)
    {
        NS_FATAL_ERROR("Data buffer too small for writeSize");
    }
    NS_LOG_INFO("MyApp Setup: Socket=" << socket << ", Peer=" << address
                                       << ", PacketSizeForScheduling=" << packetSize // Note: packetSize is passed but writeSize is used for scheduling.
                                       << ", ActualPayloadSize=" << writeSize
                                       << ", NumPackets=" << nPackets
                                       << ", DataRate=" << dataRate);
}

void MyApp::StartApplication()
{
    m_running = true;
    m_packetsSent = 0;
    NS_LOG_INFO("MyApp StartApplication: Starting at " << Simulator::Now().GetSeconds() << "s.");

    if (!m_socket)
    {
        NS_FATAL_ERROR("Socket not initialized");
    }

    m_socket->Bind();
    m_socket->Connect(m_peer);
    NS_LOG_DEBUG("MyApp StartApplication: Socket bound and connected to " << m_peer);
    // Schedule the first packet transmission
    if (m_nPackets > 0)
    {
        ScheduleTx();
    }
    else
    {
        NS_LOG_WARN("MyApp StartApplication: m_nPackets is 0. No packets to send.");
        // Optionally stop the application immediately if no packets are needed
        // Simulator::ScheduleNow(&MyApp::StopApplication, this);
    }
}

void MyApp::StopApplication()
{
    m_running = false;
    NS_LOG_INFO("MyApp StopApplication: Stopping at " << Simulator::Now().GetSeconds() << "s. Packets sent: " << m_packetsSent);

    if (m_sendEvent.IsPending())
    {
        Simulator::Cancel(m_sendEvent);
        NS_LOG_DEBUG("MyApp StopApplication: Cancelled pending send event.");
    }

    if (m_socket)
    {
        m_socket->Close();
        NS_LOG_DEBUG("MyApp StopApplication: Socket closed.");
        // Setting to nullptr here is defensive, Ptr will handle destruction
        // m_socket = nullptr;
    }
}

void MyApp::SendPacket()
{
    if (!m_running)
    {
        NS_LOG_DEBUG("MyApp SendPacket: Application not running, skipping packet send.");
        return;
    }

    if (m_packetsSent >= m_nPackets)
    {
        NS_LOG_DEBUG("MyApp SendPacket: All " << m_nPackets << " packets scheduled have been sent.");
        // This can happen if ScheduleTx is called but m_running is set to false just before SendPacket executes
        // or due to logic errors. We should not attempt to send more than m_nPackets.
        return;
    }

    Ptr<Packet> packet;
    if (m_packetsSent == m_nPackets - 1)
    {
        // packet = Create<Packet>(m_data_fin, m_writeSize);
        packet = Create<Packet>(m_data_fin.data(), m_writeSize);

        // Add FIN header
        FinHeader finHeader;
        finHeader.SetIsFin(true);
        packet->AddHeader(finHeader);

        NS_LOG_INFO("MyApp SendPacket: Sending final packet (FIN) with header");
    }
    else
    {
        packet = Create<Packet>(m_data.data(), m_writeSize);
        // packet = Create<Packet>(m_data, m_writeSize);
        NS_LOG_DEBUG("MyApp SendPacket: Sending data packet");
    }

    // --- CORRECTED ERROR HANDLING ---
    ssize_t bytesSent = m_socket->Send(packet); // Use ssize_t to get the correct return value

    if (bytesSent < 0)
    {                                                                                                                                                      // Check if the signed return value is negative
        NS_LOG_ERROR("MyApp SendPacket: Error sending packet. Socket error: " << m_socket->GetErrno() << " at " << Simulator::Now().GetSeconds() << "s."); // Log the specific socket error
        // Consider stopping the application or implementing error handling/retry logic
        // StopApplication(); // Uncomment this line if you want the application to stop on send error
    }
    else if ((uint32_t)bytesSent != m_writeSize)
    { // Compare with the expected size (casting bytesSent to uint32_t for comparison with m_writeSize which is uint64_t, ensure compatibility or cast m_writeSize if appropriate)
        // Note: A partial send might occur if the socket buffer cannot hold the entire packet.
        // Given TCP and a writeSize <= MSS, this is less common than a -1 error for buffer full.
        // The error code check above is more likely for a buffer full scenario in NS-3 TCP.
        NS_LOG_WARN("MyApp SendPacket: Sent " << bytesSent << " bytes, expected " << m_writeSize << " bytes. Possible partial send or queuing issue at " << Simulator::Now().GetSeconds() << "s.");
    }
    else
    {
        NS_LOG_DEBUG("MyApp SendPacket: Successfully sent " << bytesSent << " bytes at " << Simulator::Now().GetSeconds() << "s.");
    }
    // --- END CORRECTED ERROR HANDLING ---

    ++m_packetsSent;

    // Schedule the next packet ONLY if more packets are remaining AND the application is still running
    if (m_packetsSent < m_nPackets)
    {
        ScheduleTx();
    }
    else
    {
        NS_LOG_INFO("MyApp SendPacket: All " << m_nPackets << " packets sent for this stream. MyApp task complete at " << Simulator::Now().GetSeconds() << "s.");
        // Application should stop itself or be stopped by the simulation manager
        // Simulator::ScheduleNow(&MyApp::StopApplication, this); // Self-stop
    }
}

void MyApp::ScheduleTx()
{
    if (m_running)
    {
        // Calculate the time for the next packet based on the actual payload size (m_writeSize) and data rate
        double seconds = static_cast<double>(m_writeSize * 8) / m_dataRate.GetBitRate();
        Time tNext = Seconds(seconds);
        // Ensure the next event is scheduled only if the application is running
        m_sendEvent = Simulator::Schedule(tNext, &MyApp::SendPacket, this);
        NS_LOG_DEBUG("MyApp ScheduleTx: Next packet (" << m_packetsSent + 1 << "/" << m_nPackets << ") scheduled for " << tNext.GetSeconds() << "s from now (at " << (Simulator::Now() + tNext).GetSeconds() << "s).");
    }
    else
    {
        NS_LOG_DEBUG("MyApp ScheduleTx: Application not running, not scheduling next transmission.");
    }
}