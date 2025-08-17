"""
Integration tests for the Flask API endpoints in the FL API.
"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from api import FLAPIServer
from config import FLConfig
from training import FLSimulationState


class TestFLAPIServer:
    """Test FLAPIServer class and API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        server = FLAPIServer()
        server.app.config['TESTING'] = True
        return server.app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def server(self, app):
        """Get the server instance"""
        return app.view_functions['ping'].__self__
    
    def test_ping_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/ping')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == {"message": "pong"}
    
    def test_configure_endpoint_valid_config(self, client):
        """Test configuration with valid data"""
        config_data = {
            'dataset': 'mnist',
            'num_clients': 5,
            'clients_per_round': 3,
            'batch_size': 64,
            'client_lr': 0.01
        }
        
        response = client.post('/configure', 
                              data=json.dumps(config_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Configuration successful'
        assert 'config' in data
        assert data['config']['dataset'] == 'mnist'
        assert data['config']['num_clients'] == 5
    
    def test_configure_endpoint_invalid_config(self, client):
        """Test configuration with invalid data"""
        config_data = {
            'dataset': 'invalid_dataset',
            'num_clients': -1,
            'batch_size': 0
        }
        
        response = client.post('/configure',
                              data=json.dumps(config_data),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid configuration' in data['error']
    
    def test_configure_endpoint_no_data(self, client):
        """Test configuration with no data"""
        response = client.post('/configure',
                              data=None,
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'No configuration provided'
    
    def test_configure_endpoint_simulation_active(self, client, server):
        """Test configuration when simulation is already active"""
        # Set simulation as initialized
        server.state.simulation_initialized = True
        
        config_data = {'dataset': 'mnist'}
        
        response = client.post('/configure',
                              data=json.dumps(config_data),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Simulation already active' in data['error']
        
        # Reset state
        server.state.simulation_initialized = False
    
    @patch('api.FLCoordinator.initialize')
    def test_initialize_simulation_success(self, mock_initialize, client, server):
        """Test successful simulation initialization"""
        # First configure
        config_data = {'dataset': 'mnist', 'num_clients': 3, 'clients_per_round': 2}
        client.post('/configure',
                   data=json.dumps(config_data),
                   content_type='application/json')
        
        # Setup mocks to avoid actual initialization
        def mock_init_side_effect(config):
            server.state.history_log = {
                'global_test_loss': [0.5],
                'global_test_accuracy': [0.8]
            }
            server.state.eligible_client_indices = [0, 1, 2]
            server.state.client_num_samples = [100, 150, 80]
            server.state.simulation_initialized = True
        
        mock_initialize.side_effect = mock_init_side_effect
        
        response = client.post('/initialize_simulation')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Simulation initialized'
        assert data['num_clients_total'] == 3
        assert data['num_eligible_clients'] == 3
        assert data['client_samples_distribution'] == [100, 150, 80]
        assert 'initial_model_evaluation' in data
        
        mock_initialize.assert_called_once()
    
    def test_initialize_simulation_not_configured(self, client):
        """Test initialization without configuration"""
        response = client.post('/initialize_simulation')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Not configured'
    
    def test_initialize_simulation_already_initialized(self, client, server):
        """Test initialization when already initialized"""
        # Set as already initialized
        server.state.simulation_initialized = True
        
        response = client.post('/initialize_simulation')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Already initialized'
        
        # Reset state
        server.state.simulation_initialized = False
    
    def test_initialize_simulation_training_active(self, client, server):
        """Test initialization when training round is active"""
        server.state.is_training_round_active = True
        
        response = client.post('/initialize_simulation')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Training round active'
        
        # Reset state
        server.state.is_training_round_active = False
    
    @patch('api.FLCoordinator.run_round')
    def test_run_round_success(self, mock_run_round, client, server):
        """Test successful round execution"""
        # Setup state
        server.state.simulation_initialized = True
        server.state.is_training_round_active = False
        
        # Mock round results
        mock_round_results = {
            'round': 1,
            'selected_clients': [0, 1],
            'metrics': {
                'avg_client_loss': 0.45,
                'avg_client_accuracy': 0.82,
                'global_test_loss': 0.4,
                'global_test_accuracy': 0.85
            },
            'duration': 5.2,
            'client_performance': {
                0: {'loss': 0.5, 'accuracy': 0.8},
                1: {'loss': 0.4, 'accuracy': 0.84}
            }
        }
        mock_run_round.return_value = mock_round_results
        
        # Test without client selection
        response = client.post('/run_round')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Round 1 completed'
        assert data['round'] == 1
        assert data['selected_clients_ns3_indices'] == [0, 1]
        assert data['avg_client_loss'] == 0.45
        assert data['avg_client_accuracy'] == 0.82
        assert data['global_test_loss'] == 0.4
        assert data['global_test_accuracy'] == 0.85
        assert data['round_duration_seconds'] == 5.2
        assert 'simulated_client_performance' in data
        
        mock_run_round.assert_called_once_with(None)
        
        # Test state flags
        assert server.state.is_training_round_active is False  # Should be reset after
    
    @patch('api.FLCoordinator.run_round')
    def test_run_round_with_selected_clients(self, mock_run_round, client, server):
        """Test round execution with selected clients"""
        server.state.simulation_initialized = True
        
        mock_round_results = {
            'round': 1, 'selected_clients': [0, 2], 'metrics': {},
            'duration': 3.0, 'client_performance': {}
        }
        mock_run_round.return_value = mock_round_results
        
        # Send with selected clients
        request_data = {'client_indices': [0, 2]}
        response = client.post('/run_round',
                              data=json.dumps(request_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        mock_run_round.assert_called_once_with([0, 2])
    
    def test_run_round_not_initialized(self, client, server):
        """Test round execution when not initialized"""
        server.state.simulation_initialized = False
        
        response = client.post('/run_round')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Not initialized'
    
    def test_run_round_already_active(self, client, server):
        """Test round execution when round is already active"""
        server.state.simulation_initialized = True
        server.state.is_training_round_active = True
        
        response = client.post('/run_round')
        
        assert response.status_code == 409
        data = json.loads(response.data)
        assert data['error'] == 'Round already active'
        
        # Reset state
        server.state.is_training_round_active = False
    
    def test_get_status_endpoint(self, client, server):
        """Test status endpoint"""
        # Setup some state
        config = FLConfig(dataset='mnist', num_clients=5)
        server.state.config = config
        server.state.simulation_initialized = True
        server.state.current_round = 3
        server.state.eligible_client_indices = [0, 1, 2, 3, 4]
        server.state.client_num_samples = [100, 150, 80, 120, 90]
        server.state.history_log = {
            'round': [1, 2, 3],
            'avg_client_loss': [0.6, 0.5, 0.4],
            'global_test_accuracy': [0.75, 0.80, 0.85]
        }
        
        response = client.get('/get_status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['simulation_initialized'] is True
        assert data['current_round'] == 3
        assert data['num_total_clients'] == 5
        assert data['num_eligible_clients'] == 5
        assert data['client_data_distribution (unique_samples)'] == [100, 150, 80, 120, 90]
        assert 'configuration' in data
        assert 'training_history' in data
        
        # Check history formatting
        history = data['training_history']
        assert history['round'] == [1, 2, 3]
        assert history['avg_client_loss'] == ['0.6000', '0.5000', '0.4000']
        assert history['global_test_accuracy'] == ['0.7500', '0.8000', '0.8500']
    
    def test_get_status_not_configured(self, client, server):
        """Test status when not configured"""
        response = client.get('/get_status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['simulation_initialized'] is False
        assert data['configuration'] == "Not configured"
        assert data['num_total_clients'] == "N/A"
        assert data['num_eligible_clients'] == "N/A"
    
    @patch('api.FLCoordinator._evaluate_and_log')
    def test_evaluate_global_model_success(self, mock_evaluate, client, server):
        """Test global model evaluation"""
        server.state.simulation_initialized = True
        server.state.current_round = 2
        
        mock_evaluate.return_value = {
            'global_test_loss': 0.35,
            'global_test_accuracy': 0.88
        }
        
        response = client.get('/evaluate_global_model')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['global_test_loss'] == 0.35
        assert data['global_test_accuracy'] == 0.88
        
        mock_evaluate.assert_called_once_with(2)
    
    def test_evaluate_global_model_not_initialized(self, client, server):
        """Test evaluation when not initialized"""
        server.state.simulation_initialized = False
        
        response = client.get('/evaluate_global_model')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Not initialized'
    
    @patch('api.FLCoordinator.reset')
    def test_reset_simulation_success(self, mock_reset, client, server):
        """Test successful simulation reset"""
        server.state.is_training_round_active = False
        
        response = client.post('/reset_simulation')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Simulation reset'
        
        mock_reset.assert_called_once()
    
    def test_reset_simulation_round_active(self, client, server):
        """Test reset when round is active"""
        server.state.is_training_round_active = True
        
        response = client.post('/reset_simulation')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Cannot reset during active round'
        
        # Reset state
        server.state.is_training_round_active = False
    
    def test_get_client_info_specific_client(self, client, server):
        """Test getting specific client info"""
        # Setup state
        server.state.simulation_initialized = True
        server.state.config = FLConfig(num_clients=3)
        server.state.client_num_samples = [100, 150, 0]
        server.state.eligible_client_indices = [0, 1]
        server.state.client_metadata = {
            0: {'rounds_participated': 2, 'total_loss': 1.2},
            1: {'rounds_participated': 1, 'total_loss': 0.8},
            2: {'rounds_participated': 0, 'total_loss': 0.0}
        }
        
        response = client.get('/get_client_info?client_id=1')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['client_id'] == 1
        assert data['num_unique_samples'] == 150
        assert data['is_eligible'] is True
        assert data['metadata']['rounds_participated'] == 1
        assert data['metadata']['total_loss'] == 0.8
    
    def test_get_client_info_invalid_client_id(self, client, server):
        """Test getting info for invalid client ID"""
        server.state.simulation_initialized = True
        server.state.config = FLConfig(num_clients=3)
        
        # Test out of range
        response = client.get('/get_client_info?client_id=5')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Invalid client ID'
        
        # Test invalid format
        response = client.get('/get_client_info?client_id=abc')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Invalid client ID format'
    
    def test_get_client_info_all_clients(self, client, server):
        """Test getting all clients info"""
        server.state.simulation_initialized = True
        server.state.config = FLConfig(num_clients=2)
        server.state.client_num_samples = [100, 0]
        server.state.eligible_client_indices = [0]
        server.state.client_metadata = {
            0: {'rounds_participated': 2},
            1: {'rounds_participated': 0}
        }
        
        response = client.get('/get_client_info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'all_clients_info' in data
        
        clients_info = data['all_clients_info']
        assert len(clients_info) == 2
        
        assert clients_info[0]['client_id'] == 0
        assert clients_info[0]['num_unique_samples'] == 100
        assert clients_info[0]['is_eligible'] is True
        
        assert clients_info[1]['client_id'] == 1
        assert clients_info[1]['num_unique_samples'] == 0
        assert clients_info[1]['is_eligible'] is False
    
    def test_get_client_info_not_initialized(self, client, server):
        """Test getting client info when not initialized"""
        server.state.simulation_initialized = False
        
        response = client.get('/get_client_info')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'Not initialized'
    
    def test_get_model_weights_success(self, client, server):
        """Test getting model weights metadata"""
        server.state.current_global_weights = [
            np.array([[1, 2], [3, 4]]),  # Shape (2, 2), 4 params
            np.array([0.5, 1.5]),        # Shape (2,), 2 params
            np.array([[[1, 2, 3]]])      # Shape (1, 1, 3), 3 params
        ]
        
        response = client.get('/get_global_model_weights')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Model weights metadata'
        assert data['num_layers_with_weights'] == 3
        assert data['weight_shapes_per_layer'] == [[2, 2], [2], [1, 1, 3]]
        assert data['total_parameters'] == 9  # 4 + 2 + 3
        assert data['model_size_bytes'] == 36  # 9 * 4 bytes
    
    def test_get_model_weights_no_weights(self, client, server):
        """Test getting model weights when none available"""
        server.state.current_global_weights = None
        
        response = client.get('/get_global_model_weights')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'No model weights available'


class TestAPIIntegration:
    """Integration tests for complete API workflows"""
    
    @pytest.fixture
    def app(self):
        server = FLAPIServer()
        server.app.config['TESTING'] = True
        return server.app
    
    @pytest.fixture
    def client(self, app):
        return app.test_client()
    
    @patch('training.DataLoader')
    @patch('training.ModelFactory')
    @patch('training.ClientTrainer')
    def test_complete_fl_workflow(self, mock_client_trainer, mock_model_factory, 
                                 mock_data_loader, client):
        """Test complete federated learning workflow"""
        
        # Setup mocks for initialization
        mock_data_loader_instance = Mock()
        mock_data_loader.return_value = mock_data_loader_instance
        mock_data_loader_instance.load_and_distribute.return_value = (
            [Mock(), Mock()],  # client datasets
            [100, 150],        # client samples
            Mock(),            # test dataset
            10                 # num classes
        )
        
        mock_model = Mock()
        mock_model.get_weights.return_value = [np.array([1, 2, 3])]
        mock_model.evaluate.return_value = (0.5, 0.8)
        mock_model_factory.create_model.return_value = mock_model
        
        mock_trainer_instance = Mock()
        mock_client_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.side_effect = [
            ([np.array([2, 3, 4])], 0.4, 0.85),
            ([np.array([3, 4, 5])], 0.35, 0.88)
        ]
        
        # Step 1: Configure
        config_data = {
            'dataset': 'mnist',
            'num_clients': 2,
            'clients_per_round': 2,
            'batch_size': 32,
            'local_epochs': 1
        }
        
        response = client.post('/configure',
                              data=json.dumps(config_data),
                              content_type='application/json')
        assert response.status_code == 200
        
        # Step 2: Initialize
        with patch('training.tf.keras.utils.set_random_seed'), \
             patch('training.np.random.seed'), \
             patch('training.AggregationFactory') as mock_agg_factory:
            
            mock_agg_factory.create_aggregator.return_value = Mock()
            
            # Get the server instance and set up state properly
            server_instance = client.application.view_functions['ping'].__self__
            
            def mock_init_side_effect(config):
                server_instance.state.eligible_client_indices = [0, 1]
                server_instance.state.client_num_samples = [100, 150]
                server_instance.state.simulation_initialized = True
                server_instance.state.history_log = {
                    'global_test_loss': [0.5],
                    'global_test_accuracy': [0.8]
                }
            
            with patch.object(server_instance.coordinator, 'initialize', side_effect=mock_init_side_effect):
                response = client.post('/initialize_simulation')
                assert response.status_code == 200
                
                data = json.loads(response.data)
                assert data['message'] == 'Simulation initialized'
                assert data['num_clients_total'] == 2
                assert data['num_eligible_clients'] == 2
        
        # Step 3: Run a round
        with patch('training.np.random.choice') as mock_choice:
            mock_choice.return_value = np.array([0, 1])
            
            # Mock the coordinator's run_round method
            mock_round_results = {
                'round': 1,
                'selected_clients': [0, 1],
                'metrics': {
                    'avg_client_loss': 0.4,
                    'avg_client_accuracy': 0.85,
                    'global_test_loss': 0.35,
                    'global_test_accuracy': 0.88
                },
                'duration': 2.5,
                'client_performance': {0: {'loss': 0.4}, 1: {'loss': 0.35}}
            }
            
            def mock_run_round_side_effect(*args, **kwargs):
                server_instance.state.current_round = 1
                return mock_round_results
                
            with patch.object(server_instance.coordinator, 'run_round', side_effect=mock_run_round_side_effect):
                response = client.post('/run_round')
                assert response.status_code == 200
                
                data = json.loads(response.data)
                assert 'Round' in data['message']
                assert 'selected_clients_ns3_indices' in data
                assert 'avg_client_loss' in data
                assert 'global_test_accuracy' in data
        
        # Step 4: Check status
        response = client.get('/get_status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['simulation_initialized'] is True
        assert data['current_round'] > 0
        assert 'training_history' in data
        
        # Step 5: Evaluate model
        response = client.get('/evaluate_global_model')
        assert response.status_code == 200
        
        # Step 6: Reset
        response = client.post('/reset_simulation')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['message'] == 'Simulation reset'
    
    def test_error_handling_workflow(self, client):
        """Test error handling in various scenarios"""
        
        # Try to initialize without configuration
        response = client.post('/initialize_simulation')
        assert response.status_code == 400
        assert 'Not configured' in json.loads(response.data)['error']
        
        # Try to run round without initialization
        response = client.post('/run_round')
        assert response.status_code == 400
        assert 'Not initialized' in json.loads(response.data)['error']
        
        # Try to evaluate without initialization
        response = client.get('/evaluate_global_model')
        assert response.status_code == 400
        assert 'Not initialized' in json.loads(response.data)['error']
        
        # Configure with invalid data
        invalid_config = {'dataset': 'invalid', 'num_clients': -1}
        response = client.post('/configure',
                              data=json.dumps(invalid_config),
                              content_type='application/json')
        assert response.status_code == 400
        assert 'Invalid configuration' in json.loads(response.data)['error']
    
    def test_concurrent_request_handling(self, client):
        """Test handling of concurrent requests"""
        
        # Configure first
        config_data = {'dataset': 'mnist', 'num_clients': 2, 'clients_per_round': 2}
        response = client.post('/configure',
                              data=json.dumps(config_data),
                              content_type='application/json')
        assert response.status_code == 200
        
        # Try to configure again while already configured  
        # First set the state to indicate simulation is initialized
        server_instance = client.application.view_functions['ping'].__self__
        server_instance.state.simulation_initialized = True
        
        response = client.post('/configure',
                              data=json.dumps(config_data),
                              content_type='application/json')
        assert response.status_code == 400
        assert 'already active' in json.loads(response.data)['error']
        
        # Try to initialize multiple times
        with patch('api.FLCoordinator.initialize'):
            response1 = client.post('/initialize_simulation')
            assert response1.status_code == 200
            
            # Second call should return "Already initialized"
            response2 = client.post('/initialize_simulation')
            assert response2.status_code == 200
            assert 'Already initialized' in json.loads(response2.data)['message']