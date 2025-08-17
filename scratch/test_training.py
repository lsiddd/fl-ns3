"""
Tests for training and coordination classes in the FL API.
"""

import pytest
import numpy as np
import tensorflow as tf
import logging
import collections
from unittest.mock import Mock, patch, MagicMock

from config import FLConfig, DatasetType, OptimizerType, AggregationMethod
from training import ClientTrainer, FLSimulationState, FLCoordinator
from data import DataLoader
from models import ModelFactory, AggregationStrategy


class TestClientTrainer:
    """Test ClientTrainer class"""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def default_config(self):
        return FLConfig(
            client_optimizer='sgd',
            client_lr=0.01,
            local_epochs=1,
            batch_size=32,
            seed=42
        )
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple test model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        return model
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample TensorFlow dataset"""
        x_data = np.random.random((50, 5)).astype(np.float32)
        y_data = np.random.randint(0, 2, 50)
        
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.batch(32)
        return dataset
    
    def test_init(self, default_config, mock_logger):
        trainer = ClientTrainer(default_config, mock_logger)
        assert trainer.config == default_config
        assert trainer.logger == mock_logger
    
    def test_create_optimizer_sgd(self, mock_logger):
        config = FLConfig(client_optimizer='sgd', client_lr=0.01)
        trainer = ClientTrainer(config, mock_logger)
        
        optimizer = trainer._create_optimizer()
        
        assert isinstance(optimizer, tf.keras.optimizers.SGD)
        # Learning rate might be wrapped in a Variable, so get the value
        lr = optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr_value = lr.numpy()
        else:
            lr_value = float(lr)
        assert abs(lr_value - 0.01) < 1e-6
    
    def test_create_optimizer_adam(self, mock_logger):
        config = FLConfig(client_optimizer='adam', client_lr=0.001)
        trainer = ClientTrainer(config, mock_logger)
        
        optimizer = trainer._create_optimizer()
        
        assert isinstance(optimizer, tf.keras.optimizers.Adam)
        # Learning rate might be wrapped in a Variable, so get the value
        lr = optimizer.learning_rate
        if hasattr(lr, 'numpy'):
            lr_value = lr.numpy()
        else:
            lr_value = float(lr)
        assert abs(lr_value - 0.001) < 1e-6
    
    def test_create_optimizer_invalid(self, mock_logger):
        config = FLConfig()
        config.client_optimizer = 'invalid_optimizer'  # Set directly to bypass validation
        trainer = ClientTrainer(config, mock_logger)
        
        with pytest.raises(ValueError, match="Unknown optimizer"):
            trainer._create_optimizer()
    
    def test_train_zero_samples(self, default_config, mock_logger, sample_model):
        trainer = ClientTrainer(default_config, mock_logger)
        
        global_weights = sample_model.get_weights()
        empty_dataset = tf.data.Dataset.from_tensor_slices(([], [])).batch(1)
        
        result_weights, loss, accuracy = trainer.train(
            sample_model, global_weights, empty_dataset, num_samples=0
        )
        
        # Should return original weights and zero metrics
        assert result_weights == global_weights
        assert loss == 0.0
        assert accuracy == 0.0
        
        # Should log warning
        mock_logger.warning.assert_called_with("Client has 0 samples, skipping training")
    
    def test_train_with_data(self, default_config, mock_logger, sample_model, sample_dataset):
        trainer = ClientTrainer(default_config, mock_logger)
        
        global_weights = sample_model.get_weights()
        
        result_weights, loss, accuracy = trainer.train(
            sample_model, global_weights, sample_dataset, num_samples=50
        )
        
        # Should return updated weights
        assert len(result_weights) == len(global_weights)
        
        # Weights should be different after training (at least some layers)
        weights_changed = False
        for orig, new in zip(global_weights, result_weights):
            if not np.allclose(orig, new, atol=1e-6):
                weights_changed = True
                break
        assert weights_changed, "Weights should change after training"
        
        # Should return valid loss and accuracy
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1  # Accuracy should be between 0 and 1
    
    def test_train_model_cloning(self, default_config, mock_logger, sample_model, sample_dataset):
        """Test that training doesn't modify the original model"""
        trainer = ClientTrainer(default_config, mock_logger)
        
        original_weights = sample_model.get_weights()
        original_weights_copy = [w.copy() for w in original_weights]
        
        trainer.train(sample_model, original_weights, sample_dataset, num_samples=50)
        
        # Original model weights should be unchanged
        current_weights = sample_model.get_weights()
        for orig, curr in zip(original_weights_copy, current_weights):
            np.testing.assert_array_equal(orig, curr)
    
    def test_train_steps_calculation(self, mock_logger, sample_dataset):
        """Test that steps per epoch are calculated correctly"""
        config = FLConfig(local_epochs=2, batch_size=10)
        trainer = ClientTrainer(config, mock_logger)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(5,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        global_weights = model.get_weights()
        
        # With 50 samples, 2 epochs, batch size 10:
        # steps_per_epoch = ceil(50 * 2 / 10) = ceil(10) = 10
        
        # Create a simple model that can be compiled
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(5,)),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        global_weights = model.get_weights()
        
        with patch.object(tf.keras.Model, 'fit') as mock_fit:
            mock_fit.return_value = Mock(history={'loss': [0.5], 'sparse_categorical_accuracy': [0.8]})
            
            trainer.train(model, global_weights, sample_dataset, num_samples=50)
            
            # Check that fit was called
            mock_fit.assert_called_once()


class TestFLSimulationState:
    """Test FLSimulationState class"""
    
    def test_default_initialization(self):
        state = FLSimulationState()
        
        # Check default values
        assert state.config is None
        assert state.client_train_datasets is None
        assert state.client_num_samples is None
        assert state.centralized_test_dataset is None
        assert state.num_classes is None
        assert state.global_model is None
        assert state.current_global_weights is None
        assert state.eligible_client_indices is None
        assert isinstance(state.history_log, dict)
        assert state.current_round == 0
        assert state.simulation_initialized is False
        assert state.model_compiled is False
        assert state.data_loaded is False
        assert state.is_training_round_active is False
        assert isinstance(state.client_metadata, dict)
        assert isinstance(state.round_metrics, list)
        assert state.aggregation_strategy is None
    
    def test_custom_initialization(self):
        config = FLConfig()
        custom_history = {'custom_key': [1, 2, 3]}
        
        state = FLSimulationState(
            config=config,
            current_round=5,
            simulation_initialized=True,
            history_log=custom_history
        )
        
        assert state.config == config
        assert state.current_round == 5
        assert state.simulation_initialized is True
        assert state.history_log == custom_history


class TestFLCoordinator:
    """Test FLCoordinator class"""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def default_config(self):
        return FLConfig(
            dataset='mnist',
            num_clients=3,
            clients_per_round=2,
            batch_size=32,
            local_epochs=1,
            seed=42,
            eval_every=1
        )
    
    @pytest.fixture
    def mock_state(self):
        return FLSimulationState()
    
    def test_init(self, mock_state, mock_logger):
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        assert coordinator.state == mock_state
        assert coordinator.logger == mock_logger
        assert coordinator.data_loader is None
        assert coordinator.client_trainer is None
    
    @patch('training.DataLoader')
    @patch('training.ClientTrainer')
    @patch('training.AggregationFactory')
    @patch('training.ModelFactory')
    def test_initialize(self, mock_model_factory, mock_agg_factory, mock_client_trainer, 
                       mock_data_loader, mock_state, mock_logger, default_config):
        
        # Setup mocks
        mock_data_loader_instance = Mock()
        mock_data_loader.return_value = mock_data_loader_instance
        
        mock_client_trainer_instance = Mock()
        mock_client_trainer.return_value = mock_client_trainer_instance
        
        mock_aggregator = Mock()
        mock_agg_factory.create_aggregator.return_value = mock_aggregator
        
        mock_model = Mock()
        mock_model.get_weights.return_value = [np.array([1, 2, 3])]
        mock_model_factory.create_model.return_value = mock_model
        
        # Mock data loading
        client_datasets = [Mock(), Mock(), Mock()]
        client_samples = [100, 150, 80]
        test_dataset = Mock()
        num_classes = 10
        
        mock_data_loader_instance.load_and_distribute.return_value = (
            client_datasets, client_samples, test_dataset, num_classes
        )
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        with patch('training.tf.keras.utils.set_random_seed'), \
             patch('training.np.random.seed'), \
             patch.object(coordinator, '_evaluate_and_log') as mock_eval:
            
            coordinator.initialize(default_config)
        
        # Verify state was updated
        assert mock_state.config == default_config
        assert mock_state.client_train_datasets == client_datasets
        assert mock_state.client_num_samples == client_samples
        assert mock_state.centralized_test_dataset == test_dataset
        assert mock_state.num_classes == num_classes
        assert mock_state.global_model == mock_model
        assert mock_state.simulation_initialized is True
        assert mock_state.model_compiled is True
        assert mock_state.data_loaded is True
        assert mock_state.current_round == 0
        
        # Verify eligible clients (all have samples > 0)
        assert mock_state.eligible_client_indices == [0, 1, 2]
        
        # Verify client metadata
        assert len(mock_state.client_metadata) == 3
        for i in range(3):
            metadata = mock_state.client_metadata[i]
            assert metadata['num_samples'] == client_samples[i]
            assert metadata['is_eligible'] == (i in [0, 1, 2])
            assert metadata['rounds_participated'] == 0
        
        # Verify evaluation was called
        mock_eval.assert_called_once_with(round_num=0)
    
    def test_run_round_not_initialized(self, mock_state, mock_logger):
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            coordinator.run_round()
    
    def test_run_round_no_eligible_clients(self, mock_state, mock_logger, default_config):
        mock_state.simulation_initialized = True
        mock_state.config = default_config
        mock_state.eligible_client_indices = []
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        result = coordinator.run_round()
        
        assert result["message"] == "No eligible clients"
        assert result["round"] == 0
        mock_logger.warning.assert_called_with("No eligible clients for training")
    
    @patch.object(FLCoordinator, '_train_clients')
    @patch.object(FLCoordinator, '_calculate_round_metrics')
    @patch.object(FLCoordinator, '_evaluate_and_log')
    @patch.object(FLCoordinator, '_update_history')
    def test_run_round_success(self, mock_update_history, mock_evaluate, 
                              mock_calc_metrics, mock_train_clients,
                              mock_state, mock_logger, default_config):
        
        # Setup state
        mock_state.simulation_initialized = True
        mock_state.config = default_config
        mock_state.eligible_client_indices = [0, 1, 2]
        mock_state.current_round = 0
        
        # Setup aggregation strategy
        mock_aggregator = Mock()
        mock_aggregator.aggregate.return_value = [np.array([1, 2, 3])]
        mock_state.aggregation_strategy = mock_aggregator
        
        # Setup global model
        mock_model = Mock()
        mock_state.global_model = mock_model
        
        # Setup mocks
        round_results = {
            'client_weights': [[np.array([1, 2, 3])], [np.array([2, 3, 4])]],
            'client_samples': [100, 150],
            'client_losses': [0.5, 0.4],
            'client_accuracies': [0.8, 0.85],
            'client_performance': {0: {'loss': 0.5}, 1: {'loss': 0.4}}
        }
        mock_train_clients.return_value = round_results
        
        round_metrics = {'avg_client_loss': 0.45, 'avg_client_accuracy': 0.825}
        mock_calc_metrics.return_value = round_metrics
        
        test_metrics = {'global_test_loss': 0.3, 'global_test_accuracy': 0.9}
        mock_evaluate.return_value = test_metrics
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        with patch('training.np.random.choice') as mock_choice:
            mock_choice.return_value = np.array([0, 1])
            
            result = coordinator.run_round()
        
        # Verify round number incremented
        assert mock_state.current_round == 1
        
        # Verify client selection
        mock_choice.assert_called_once()
        
        # Verify training was called
        mock_train_clients.assert_called_once()
        # Get the actual arguments passed to _train_clients
        actual_args = mock_train_clients.call_args[0][0]
        # Convert to list for comparison
        actual_clients = list(actual_args)
        assert actual_clients == [0, 1]
        
        # Verify aggregation
        mock_aggregator.aggregate.assert_called_once_with(
            round_results['client_weights'],
            round_results['client_samples']
        )
        
        # Verify model weights updated
        mock_model.set_weights.assert_called_once()
        # Check the arguments more carefully
        call_args = mock_model.set_weights.call_args[0][0]
        assert len(call_args) == 1
        np.testing.assert_array_equal(call_args[0], np.array([1, 2, 3]))
        
        # Verify evaluation called (eval_every=1)
        mock_evaluate.assert_called_once_with(1)
        
        # Verify history updated
        expected_metrics = {**round_metrics, **test_metrics}
        mock_update_history.assert_called_once_with(1, expected_metrics)
        
        # Verify return value
        assert result['round'] == 1
        assert result['selected_clients'] == [0, 1]
        assert 'duration' in result
        assert result['client_performance'] == round_results['client_performance']
    
    def test_run_round_with_selected_clients(self, mock_state, mock_logger, default_config):
        """Test run_round with pre-selected clients"""
        
        # Setup state
        mock_state.simulation_initialized = True
        mock_state.config = default_config
        mock_state.eligible_client_indices = [0, 1, 2]
        mock_state.current_round = 0
        
        mock_aggregator = Mock()
        mock_aggregator.aggregate.return_value = None  # No weights to aggregate
        mock_state.aggregation_strategy = mock_aggregator
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        with patch.object(coordinator, '_train_clients') as mock_train_clients, \
             patch.object(coordinator, '_calculate_round_metrics') as mock_calc_metrics, \
             patch.object(coordinator, '_evaluate_and_log') as mock_evaluate, \
             patch.object(coordinator, '_update_history') as mock_update_history:
            
            mock_train_clients.return_value = {
                'client_weights': [],
                'client_samples': [],
                'client_losses': [],
                'client_accuracies': [],
                'client_performance': {}
            }
            mock_calc_metrics.return_value = {}
            mock_evaluate.return_value = {}
            
            # Test with valid selected clients
            result = coordinator.run_round(selected_clients=[0, 2])
            
            # Should use the selected clients
            mock_train_clients.assert_called_once_with([0, 2])
            assert result['selected_clients'] == [0, 2]
    
    def test_run_round_invalid_selected_clients(self, mock_state, mock_logger, default_config):
        """Test run_round with some invalid selected clients"""
        
        # Setup state
        mock_state.simulation_initialized = True
        mock_state.config = default_config
        mock_state.eligible_client_indices = [0, 1]  # Only 0 and 1 are eligible
        mock_state.current_round = 0
        
        mock_aggregator = Mock()
        mock_aggregator.aggregate.return_value = None
        mock_state.aggregation_strategy = mock_aggregator
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        with patch.object(coordinator, '_train_clients') as mock_train_clients, \
             patch.object(coordinator, '_calculate_round_metrics') as mock_calc_metrics, \
             patch.object(coordinator, '_evaluate_and_log') as mock_evaluate, \
             patch.object(coordinator, '_update_history') as mock_update_history:
            
            mock_train_clients.return_value = {
                'client_weights': [], 'client_samples': [], 'client_losses': [],
                'client_accuracies': [], 'client_performance': {}
            }
            mock_calc_metrics.return_value = {}
            mock_evaluate.return_value = {}
            
            # Test with some invalid clients (2 is not eligible)
            result = coordinator.run_round(selected_clients=[0, 2])
            
            # Should filter out invalid clients and warn
            mock_train_clients.assert_called_once_with([0])
            mock_logger.warning.assert_called()
            assert result['selected_clients'] == [0]
    
    @patch.object(FLCoordinator, '_estimate_model_size')
    @patch.object(FLCoordinator, '_estimate_training_time')
    def test_train_clients(self, mock_estimate_time, mock_estimate_size,
                          mock_state, mock_logger, default_config):
        
        # Setup state
        mock_state.client_train_datasets = [Mock(), Mock()]
        mock_state.client_num_samples = [100, 150]
        mock_state.global_model = Mock()
        mock_state.current_global_weights = [np.array([1, 2, 3])]
        mock_state.client_metadata = {
            0: {'rounds_participated': 0, 'total_loss': 0.0, 'total_accuracy': 0.0},
            1: {'rounds_participated': 0, 'total_loss': 0.0, 'total_accuracy': 0.0}
        }
        
        # Setup client trainer
        mock_trainer = Mock()
        mock_trainer.train.side_effect = [
            ([np.array([2, 3, 4])], 0.5, 0.8),  # Client 0 results
            ([np.array([3, 4, 5])], 0.4, 0.85)  # Client 1 results
        ]
        
        # Setup estimates
        mock_estimate_size.return_value = 1000
        mock_estimate_time.side_effect = [500, 600]
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        coordinator.client_trainer = mock_trainer
        
        result = coordinator._train_clients([0, 1])
        
        # Verify trainer was called for each client
        assert mock_trainer.train.call_count == 2
        
        # Verify results structure
        assert len(result['client_weights']) == 2
        assert len(result['client_samples']) == 2
        assert len(result['client_losses']) == 2
        assert len(result['client_accuracies']) == 2
        assert len(result['client_performance']) == 2
        
        # Verify specific values
        assert result['client_samples'] == [100, 150]
        assert result['client_losses'] == [0.5, 0.4]
        assert result['client_accuracies'] == [0.8, 0.85]
        
        # Verify client performance data
        assert result['client_performance'][0]['loss'] == 0.5
        assert result['client_performance'][0]['accuracy'] == 0.8
        assert result['client_performance'][0]['num_samples'] == 100
        assert result['client_performance'][0]['model_size_bytes'] == 1000
        assert result['client_performance'][0]['training_time_ms'] == 500
    
    def test_train_clients_zero_samples(self, mock_state, mock_logger):
        """Test training clients with zero samples"""
        
        # Setup state with client having zero samples
        mock_state.client_train_datasets = [Mock()]
        mock_state.client_num_samples = [0]
        mock_state.client_metadata = {0: {}}
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        result = coordinator._train_clients([0])
        
        # Should skip client with zero samples
        assert len(result['client_weights']) == 0
        assert len(result['client_samples']) == 0
        assert len(result['client_losses']) == 0
        assert len(result['client_accuracies']) == 0
        mock_logger.warning.assert_called_with("Client 0 has no data")
    
    def test_calculate_round_metrics(self, mock_state, mock_logger):
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        # Test with weighted average
        round_results = {
            'client_samples': [100, 200],
            'client_losses': [0.5, 0.3],
            'client_accuracies': [0.8, 0.9]
        }
        
        metrics = coordinator._calculate_round_metrics(round_results)
        
        # Weighted average: (100*0.5 + 200*0.3) / 300 = (50 + 60) / 300 = 0.367
        expected_loss = (100 * 0.5 + 200 * 0.3) / 300
        expected_acc = (100 * 0.8 + 200 * 0.9) / 300
        
        assert abs(metrics['avg_client_loss'] - expected_loss) < 1e-10
        assert abs(metrics['avg_client_accuracy'] - expected_acc) < 1e-10
    
    def test_calculate_round_metrics_no_samples(self, mock_state, mock_logger):
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        # Test with zero samples
        round_results = {
            'client_samples': [0, 0],
            'client_losses': [0.5, 0.3],
            'client_accuracies': [0.8, 0.9]
        }
        
        metrics = coordinator._calculate_round_metrics(round_results)
        
        # Should use simple average when no samples
        assert abs(metrics['avg_client_loss'] - 0.4) < 1e-10  # (0.5 + 0.3) / 2
        assert abs(metrics['avg_client_accuracy'] - 0.85) < 1e-10  # (0.8 + 0.9) / 2
    
    def test_evaluate_and_log(self, mock_state, mock_logger):
        # Setup state
        mock_model = Mock()
        mock_model.evaluate.return_value = (0.25, 0.92)
        mock_state.global_model = mock_model
        mock_state.centralized_test_dataset = Mock()
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        result = coordinator._evaluate_and_log(round_num=5)
        
        # Verify evaluation was called
        mock_model.evaluate.assert_called_once_with(
            mock_state.centralized_test_dataset, verbose=0
        )
        
        # Verify results
        assert result['global_test_loss'] == 0.25
        assert result['global_test_accuracy'] == 0.92
        
        # Verify logging
        mock_logger.info.assert_any_call("Evaluating global model (round 5)")
        mock_logger.info.assert_any_call("Test Loss: 0.2500, Test Accuracy: 0.9200")
    
    def test_evaluate_and_log_no_model(self, mock_state, mock_logger):
        # Setup state without model
        mock_state.global_model = None
        mock_state.centralized_test_dataset = None
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        result = coordinator._evaluate_and_log(round_num=1)
        
        # Should return NaN values
        assert np.isnan(result['global_test_loss'])
        assert np.isnan(result['global_test_accuracy'])
    
    def test_update_history(self, mock_state, mock_logger):
        mock_state.history_log = collections.defaultdict(list)
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        metrics = {
            'avg_client_loss': 0.4,
            'avg_client_accuracy': 0.85,
            'global_test_loss': 0.3,
            'global_test_accuracy': 0.9,
            'extra_metric': 'ignored'
        }
        
        coordinator._update_history(round_num=3, metrics=metrics)
        
        # Verify history was updated
        assert mock_state.history_log['round'] == [3]
        assert mock_state.history_log['avg_client_loss'] == [0.4]
        assert mock_state.history_log['avg_client_accuracy'] == [0.85]
        assert mock_state.history_log['global_test_loss'] == [0.3]
        assert mock_state.history_log['global_test_accuracy'] == [0.9]
        
        # Extra metrics should be ignored
        assert 'extra_metric' not in mock_state.history_log
    
    def test_estimate_model_size(self, mock_state, mock_logger):
        # Setup weights
        mock_state.current_global_weights = [
            np.array([[1, 2], [3, 4]]),  # 4 parameters
            np.array([0.5, 1.5]),        # 2 parameters
            np.array([[[1]]])            # 1 parameter
        ]
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        size = coordinator._estimate_model_size()
        
        # Total: 4 + 2 + 1 = 7 parameters, 7 * 4 bytes = 28 bytes
        assert size == 28
    
    def test_estimate_model_size_no_weights(self, mock_state, mock_logger):
        mock_state.current_global_weights = None
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        size = coordinator._estimate_model_size()
        
        assert size == 0
    
    def test_estimate_training_time(self, mock_state, mock_logger):
        mock_state.config = FLConfig(local_epochs=2)
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        time_ms = coordinator._estimate_training_time(num_samples=100)
        
        # Base time (100) + 100 samples * 0.5 ms/sample * 2 epochs = 100 + 100 = 200
        expected_time = 100 + 100 * 0.5 * 2
        assert time_ms == expected_time
    
    def test_reset(self, mock_state, mock_logger):
        # Setup state with values
        mock_state.config = FLConfig()
        mock_state.simulation_initialized = True
        mock_state.current_round = 5
        mock_state.history_log = {'some': ['data']}
        
        coordinator = FLCoordinator(mock_state, mock_logger)
        
        with patch('training.tf.keras.backend.clear_session') as mock_clear:
            coordinator.reset()
        
        # Verify all state was reset
        assert mock_state.config is None
        assert mock_state.simulation_initialized is False
        assert mock_state.current_round == 0
        assert isinstance(mock_state.history_log, collections.defaultdict)
        assert len(mock_state.history_log) == 0
        
        # Verify TensorFlow session was cleared
        mock_clear.assert_called_once()