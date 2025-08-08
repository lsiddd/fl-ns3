#include "ApiClient.h"
#include "ns3/log.h"
#include <iostream>

NS_LOG_COMPONENT_DEFINE("FlApiClient");

// Função auxiliar para extrair host e porta da URL
std::pair<std::string, int> parse_url(const std::string& url) {
    std::string host;
    int port = 80; // Porta padrão para HTTP
    
    size_t host_start = url.find("://");
    if (host_start != std::string::npos) {
        host_start += 3; // Pula "://"
    } else {
        host_start = 0;
    }

    size_t port_start = url.find(":", host_start);
    if (port_start != std::string::npos) {
        host = url.substr(host_start, port_start - host_start);
        try {
            port = std::stoi(url.substr(port_start + 1));
        } catch (const std::exception& e) {
            NS_LOG_ERROR("URL Invalida, falha ao analisar porta: " << url);
            port = -1; // Indica erro
        }
    } else {
        host = url.substr(host_start);
    }
    return {host, port};
}


FlApiClient::FlApiClient(const std::string& base_url) : base_url_(base_url) {
    auto host_port = parse_url(base_url);
    host_ = host_port.first;
    port_ = host_port.second;
    NS_LOG_INFO("Cliente API inicializado. Host: " << host_ << ", Porta: " << port_);
}

int FlApiClient::configure(const json& config_payload) {
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(5); // 5 segundos para conectar
    cli.set_read_timeout(20);      // 20 segundos para ler a resposta

    NS_LOG_INFO("Enviando POST para /configure");
    auto res = cli.Post("/configure", config_payload.dump(), "application/json");

    if (res) {
        NS_LOG_INFO("Resposta de /configure recebida. Status: " << res->status);
        if (res->status != 200) {
            NS_LOG_ERROR("Erro em /configure. Resposta: " << res->body);
        }
        return res->status;
    } else {
        auto err = res.error();
        NS_LOG_ERROR("Falha na chamada para /configure: " << httplib::to_string(err));
        return -1; // Retorna -1 para indicar falha na comunicação
    }
}

int FlApiClient::initializeSimulation() {
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(60); // A inicialização pode demorar mais

    NS_LOG_INFO("Enviando POST para /initialize_simulation");
    auto res = cli.Post("/initialize_simulation");

    if (res) {
        NS_LOG_INFO("Resposta de /initialize_simulation recebida. Status: " << res->status);
        if (res->status != 200) {
            NS_LOG_ERROR("Erro em /initialize_simulation. Resposta: " << res->body);
        }
        return res->status;
    } else {
        auto err = res.error();
        NS_LOG_ERROR("Falha na chamada para /initialize_simulation: " << httplib::to_string(err));
        return -1;
    }
}

std::pair<int, json> FlApiClient::runRound(const json& clients_payload) {
    httplib::Client cli(host_, port_);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(120); // A rodada de treinamento pode ser longa

    NS_LOG_INFO("Enviando POST para /run_round");
    auto res = cli.Post("/run_round", clients_payload.dump(), "application/json");

    json response_body;
    if (res) {
        NS_LOG_INFO("Resposta de /run_round recebida. Status: " << res->status);
        try {
            if (!res->body.empty()) {
                response_body = json::parse(res->body);
            }
        } catch (json::parse_error& e) {
            NS_LOG_ERROR("Falha ao analisar resposta JSON de /run_round: " << e.what());
            // Retorna o status HTTP, mas com um JSON de erro
            return {res->status, {{"parse_error", e.what()}}};
        }
        return {res->status, response_body};
    } else {
        auto err = res.error();
        NS_LOG_ERROR("Falha na chamada para /run_round: " << httplib::to_string(err));
        return {-1, {{"connection_error", httplib::to_string(err)}}};
    }
}
