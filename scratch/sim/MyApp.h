#pragma once

#include "ns3/application.h"
#include "ns3/socket.h"
#include "ns3/address.h"
#include "ns3/data-rate.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"

using namespace ns3;

class FinHeader : public Header {
public:
    FinHeader() : m_isFin(false) {}
    
    static TypeId GetTypeId();
    TypeId GetInstanceTypeId() const override;
    uint32_t GetSerializedSize() const override;
    void Serialize(Buffer::Iterator start) const override;
    uint32_t Deserialize(Buffer::Iterator start) override;
    void Print(std::ostream &os) const override;
    
    void SetIsFin(bool isFin) { m_isFin = isFin; }
    bool IsFin() const { return m_isFin; }

private:
    bool m_isFin;
};

class MyApp : public Application {
public:
    MyApp();
    virtual ~MyApp() override;

    // Setup method now explicitly notes that packetSize is for scheduling logic (even if writeSize is used now)
    // and writeSize is the actual payload size.
    void Setup(Ptr<Socket> socket, Address address, uint32_t packetSize,
               uint32_t nPackets, DataRate dataRate, uint32_t writeSize, uint8_t* data, uint8_t* dataFin);
    virtual void StopApplication() override;

private:
    virtual void StartApplication() override;
    void ScheduleTx();
    void SendPacket();

    Ptr<Socket> m_socket{nullptr};
    Address m_peer;
    // uint32_t m_packetSize{0}; // Removing m_packetSize as it's not used for scheduling or sending payload size calculation
    uint64_t m_nPackets{0};
    DataRate m_dataRate;
    EventId m_sendEvent;
    bool m_running{false};
    uint64_t m_packetsSent{0};
    Time m_startTime; // Stores the app start time
    uint8_t* m_data; // Pointer to the normal data buffer (owned by caller)
    uint8_t* m_data_fin; // Pointer to the final packet data buffer (owned by caller)
    uint64_t m_writeSize; // Actual size of payload for each packet

};
