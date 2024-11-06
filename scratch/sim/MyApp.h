#ifndef MY_APP_H
#define MY_APP_H

#include "ns3/application.h"
#include "ns3/socket.h"
#include "ns3/address.h"
#include "ns3/data-rate.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"

using namespace ns3;

class MyApp : public Application
{
public:
    MyApp();
    virtual ~MyApp() override;

    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
               uint32_t nPackets, DataRate dataRate, uint32_t writeSize, uint8_t* data, uint8_t* dataFin);
    virtual void StopApplication() override;

private:
    virtual void StartApplication() override;
    void ScheduleTx();
    void SendPacket();

    Ptr<Socket> m_socket{nullptr};
    Address m_peer;
    uint32_t m_packetSize{0};
    uint32_t m_nPackets{0};
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running{false};
    uint32_t m_packetsSent{0};
    Time m_startTime;
    uint8_t* m_data;
    uint8_t* m_data_fin;
    uint32_t m_writeSize;
};

#endif

