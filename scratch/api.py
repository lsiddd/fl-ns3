"""
Flask API server for Federated Learning simulation.
"""

import os
import numpy as np
import logging
import socket
from flask import Flask, request, jsonify
from dataclasses import asdict

# Set TensorFlow environment variable before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from config import FLConfig
from training import FLSimulationState, FLCoordinator


class FLAPIServer:
    """Flask API server for federated learning"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.state = FLSimulationState()
        self.coordinator = FLCoordinator(self.state, self._setup_logger())
        self._setup_routes()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        self.app.route('/ping', methods=['GET'])(self.ping)
        self.app.route('/configure', methods=['POST'])(self.configure)
        self.app.route('/initialize_simulation', methods=['POST'])(self.initialize_simulation)
        self.app.route('/run_round', methods=['POST'])(self.run_round)
        self.app.route('/get_status', methods=['GET'])(self.get_status)
        self.app.route('/evaluate_global_model', methods=['GET'])(self.evaluate_global_model)
        self.app.route('/reset_simulation', methods=['POST'])(self.reset_simulation)
        self.app.route('/get_client_info', methods=['GET'])(self.get_client_info)
        self.app.route('/get_global_model_weights', methods=['GET'])(self.get_model_weights)
    
    def ping(self):
        """Health check endpoint"""
        return jsonify({"message": "pong"}), 200
    
    def configure(self):
        """Configure FL simulation"""
        
        if self.state.simulation_initialized or self.state.is_training_round_active:
            return jsonify({
                "error": "Simulation already active. Reset first."
            }), 400
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No configuration provided"}), 400
            
            # Create config from request
            config = FLConfig.from_dict(data)
            
            # Validate config
            valid, error = config.validate()
            if not valid:
                return jsonify({"error": f"Invalid configuration: {error}"}), 400
            
            self.state.config = config
            
            return jsonify({
                "message": "Configuration successful",
                "config": asdict(config)
            }), 200
            
        except Exception as e:
            self.coordinator.logger.exception(f"Configuration error: {e}")
            return jsonify({"error": str(e)}), 500
    
    def initialize_simulation(self):
        """Initialize FL simulation"""
        
        if self.state.is_training_round_active:
            return jsonify({"error": "Training round active"}), 400
        
        if self.state.simulation_initialized:
            return jsonify({"message": "Already initialized"}), 200
        
        if not self.state.config:
            return jsonify({"error": "Not configured"}), 400
        
        try:
            self.coordinator.initialize(self.state.config)
            
            # Get initial evaluation
            initial_metrics = {}
            if self.state.history_log['global_test_loss']:
                initial_metrics = {
                    'loss': self.state.history_log['global_test_loss'][0],
                    'accuracy': self.state.history_log['global_test_accuracy'][0]
                }
            
            return jsonify({
                "message": "Simulation initialized",
                "num_clients_total": self.state.config.num_clients,
                "num_eligible_clients": len(self.state.eligible_client_indices),
                "client_samples_distribution": self.state.client_num_samples,
                "initial_model_evaluation": initial_metrics
            }), 200
            
        except Exception as e:
            self.coordinator.logger.exception(f"Initialization error: {e}")
            self.state.simulation_initialized = False
            return jsonify({"error": str(e)}), 500
    
    def run_round(self):
        """Run one training round"""
        
        if self.state.is_training_round_active:
            return jsonify({"error": "Round already active"}), 409
        
        if not self.state.simulation_initialized:
            return jsonify({"error": "Not initialized"}), 400
        
        self.state.is_training_round_active = True
        
        try:
            data = request.get_json(silent=True) or {}
            selected_clients = data.get('client_indices', None)
            
            # Run round
            results = self.coordinator.run_round(selected_clients)
            
            # Format response
            response = {
                "message": f"Round {results['round']} completed",
                "round": results['round'],
                "selected_clients_ns3_indices": results['selected_clients'],
                "avg_client_loss": results['metrics'].get('avg_client_loss', float('nan')),
                "avg_client_accuracy": results['metrics'].get('avg_client_accuracy', float('nan')),
                "global_test_loss": results['metrics'].get('global_test_loss', float('nan')),
                "global_test_accuracy": results['metrics'].get('global_test_accuracy', float('nan')),
                "round_duration_seconds": results['duration'],
                "simulated_client_performance": results['client_performance']
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            self.coordinator.logger.exception(f"Round execution error: {e}")
            return jsonify({"error": str(e)}), 500
            
        finally:
            self.state.is_training_round_active = False
    
    def get_status(self):
        """Get simulation status"""
        
        config_dict = asdict(self.state.config) if self.state.config else "Not configured"
        
        # Format history
        history = {}
        if self.state.history_log:
            for key, values in self.state.history_log.items():
                if key in ['avg_client_loss', 'avg_client_accuracy', 
                          'global_test_loss', 'global_test_accuracy']:
                    history[key] = [
                        f"{v:.4f}" if not np.isnan(v) else "NaN" 
                        for v in values
                    ]
                else:
                    history[key] = values
        
        status = {
            "simulation_initialized": self.state.simulation_initialized,
            "data_loaded": self.state.data_loaded,
            "model_compiled": self.state.model_compiled,
            "current_round": self.state.current_round,
            "is_training_round_active": self.state.is_training_round_active,
            "configuration": config_dict,
            "num_total_clients": self.state.config.num_clients if self.state.config else "N/A",
            "num_eligible_clients": len(self.state.eligible_client_indices) if self.state.eligible_client_indices else "N/A",
            "client_data_distribution (unique_samples)": self.state.client_num_samples,
            "training_history": history
        }
        
        return jsonify(status), 200
    
    def evaluate_global_model(self):
        """Evaluate global model on demand"""
        
        if not self.state.simulation_initialized:
            return jsonify({"error": "Not initialized"}), 400
        
        try:
            metrics = self.coordinator._evaluate_and_log(self.state.current_round)
            return jsonify(metrics), 200
            
        except Exception as e:
            self.coordinator.logger.exception(f"Evaluation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    def reset_simulation(self):
        """Reset simulation"""
        
        if self.state.is_training_round_active:
            return jsonify({"error": "Cannot reset during active round"}), 400
        
        self.coordinator.reset()
        
        return jsonify({"message": "Simulation reset"}), 200
    
    def get_client_info(self):
        """Get client information"""
        
        if not self.state.simulation_initialized:
            return jsonify({"error": "Not initialized"}), 400
        
        client_id = request.args.get('client_id')
        
        if client_id:
            try:
                client_id = int(client_id)
                
                if not (0 <= client_id < self.state.config.num_clients):
                    return jsonify({"error": "Invalid client ID"}), 400
                
                info = {
                    "client_id": client_id,
                    "num_unique_samples": self.state.client_num_samples[client_id],
                    "is_eligible": client_id in self.state.eligible_client_indices,
                    "metadata": self.state.client_metadata.get(client_id, {})
                }
                
                return jsonify(info), 200
                
            except ValueError:
                return jsonify({"error": "Invalid client ID format"}), 400
        else:
            # Return all clients info
            all_info = []
            for i in range(self.state.config.num_clients):
                all_info.append({
                    "client_id": i,
                    "num_unique_samples": self.state.client_num_samples[i],
                    "is_eligible": i in self.state.eligible_client_indices,
                    "metadata": self.state.client_metadata.get(i, {})
                })
            
            return jsonify({"all_clients_info": all_info}), 200
    
    def get_model_weights(self):
        """Get global model weights metadata"""
        
        if not self.state.current_global_weights:
            return jsonify({"error": "No model weights available"}), 400
        
        try:
            num_layers = len(self.state.current_global_weights)
            shapes = [list(w.shape) for w in self.state.current_global_weights]
            total_params = sum(w.size for w in self.state.current_global_weights)
            
            return jsonify({
                "message": "Model weights metadata",
                "num_layers_with_weights": num_layers,
                "weight_shapes_per_layer": shapes,
                "total_parameters": int(total_params),
                "model_size_bytes": int(total_params * 4)  # float32
            }), 200
            
        except Exception as e:
            self.coordinator.logger.exception(f"Error getting model weights: {e}")
            return jsonify({"error": str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server"""
        
        # Try to find available port
        start_port = port
        port_range = 100
        
        for p in range(start_port, start_port + port_range):
            try:
                # Test if port is available
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.bind((host, p))
                test_socket.close()
                
                # Port is available
                print(f"FL_API_PORT:{p}")
                self.coordinator.logger.info(f"Starting server on {host}:{p}")
                
                self.app.run(host=host, port=p, debug=debug, threaded=False)
                break
                
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    continue
                else:
                    raise
        else:
            self.coordinator.logger.critical(f"No available port in range {start_port}-{start_port+port_range}")
            exit(1)