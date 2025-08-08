#pragma once

#include "httplib.h" // A nova biblioteca de cliente HTTP
#include "json.hpp"  // A biblioteca JSON existente
#include <string>
#include <utility> // Para std::pair

using json = nlohmann::json;

class FlApiClient
{
public:
    // O construtor recebe a URL base do servidor da API
    FlApiClient(const std::string& base_url);

    // Chama o endpoint /configure da API
    int configure(const json& config_payload);

    // Chama o endpoint /initialize_simulation da API
    int initializeSimulation();

    // Chama o endpoint /run_round da API
    // Retorna o código de status HTTP e o corpo da resposta JSON
    std::pair<int, json> runRound(const json& clients_payload);

private:
    std::string base_url_;
    // Remove a parte do esquema (http://) para o construtor do httplib
    std::string host_;
    int port_;
};
