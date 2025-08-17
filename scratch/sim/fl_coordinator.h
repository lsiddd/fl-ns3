#pragma once

#include "client_types.h"
#include "dataframe.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include <string>
#include <vector>

using namespace ns3;

extern DataFrame accuracy_df;
extern DataFrame participation_df;
extern std::vector<ClientModels> clientsInfoGlobal;
extern std::vector<ClientModels> selectedClientsForCurrentRound;
extern std::vector<NodesIps> nodesIPs;

class FLCoordinator {
public:
    static bool initializeFlApi();
    static void selectNs3ManagedClients(int n_to_select);
    static bool triggerAndProcessFLRoundInApi();
    static void sendModelsToServer();
    static bool isRoundTimedOut(Time roundStartTimeNs3Comms);
    static void logRoundTimeout();
    static void addParticipationToDataFrame();
    static void finalizeNs3CommsPhase();
    static void startNewFLRound(Time &roundStartTimeNs3CommsParam);
    static void manager();

private:
    static std::string getEnvVar(const std::string &key, const std::string &default_val);
    static int callPythonApi(const std::string &endpoint, const std::string &method = "POST", 
                            const std::string &data = "", const std::string &output_file = "");
    static std::string exec(const char* cmd);
};

extern std::string FL_API_BASE_URL;
extern const int FL_API_NUM_CLIENTS;
extern const int FL_API_CLIENTS_PER_ROUND;
extern Time timeout;
extern int roundNumber;