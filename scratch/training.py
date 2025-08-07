"""
Client training and FL coordination for Federated Learning simulation.
"""

import collections
import time
import numpy as np
import tensorflow as tf
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import FLConfig, DatasetType, OptimizerType, AggregationMethod
from data import DataLoader
from models import ModelFactory, AggregationStrategy, AggregationFactory


class ClientTrainer:
    """Handles client-side training"""
    
    def __init__(self, config: FLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def train(self, model_template: tf.keras.Model, 
             global_weights: List[np.ndarray],
             client_dataset: tf.data.Dataset,
             num_samples: int) -> Tuple[List[np.ndarray], float, float]:
        """Train a client model"""
        
        if num_samples == 0:
            self.logger.warning("Client has 0 samples, skipping training")
            return global_weights, 0.0, 0.0
        
        # Clone model and set weights
        client_model = tf.keras.models.clone_model(model_template)
        
        # Create optimizer
        optimizer = self._create_optimizer()
        
        # Compile model
        client_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        # Set global weights
        client_model.set_weights(global_weights)
        
        # Calculate steps
        steps_per_epoch = int(np.ceil(
            num_samples * self.config.local_epochs / self.config.batch_size
        ))
        
        # Train
        history = client_model.fit(
            client_dataset,
            epochs=1,
            verbose=0,
            steps_per_epoch=steps_per_epoch
        )
        
        # Extract metrics
        loss = history.history.get('loss', [0.0])[-1]
        accuracy = history.history.get('sparse_categorical_accuracy', [0.0])[-1]
        
        return client_model.get_weights(), loss, accuracy
    
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer based on configuration"""
        opt_type = OptimizerType(self.config.client_optimizer)
        
        if opt_type == OptimizerType.SGD:
            return tf.keras.optimizers.SGD(learning_rate=self.config.client_lr)
        elif opt_type == OptimizerType.ADAM:
            return tf.keras.optimizers.Adam(learning_rate=self.config.client_lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.client_optimizer}")


@dataclass
class FLSimulationState:
    """Complete state of the FL simulation"""
    config: Optional[FLConfig] = None
    client_train_datasets: Optional[List[tf.data.Dataset]] = None
    client_num_samples: Optional[List[int]] = None
    centralized_test_dataset: Optional[tf.data.Dataset] = None
    num_classes: Optional[int] = None
    global_model: Optional[tf.keras.Model] = None
    current_global_weights: Optional[List[np.ndarray]] = None
    eligible_client_indices: Optional[List[int]] = None
    history_log: Dict[str, List] = field(default_factory=lambda: collections.defaultdict(list))
    current_round: int = 0
    simulation_initialized: bool = False
    model_compiled: bool = False
    data_loaded: bool = False
    is_training_round_active: bool = False
    
    # Additional realistic FL state
    client_metadata: Dict[int, Dict] = field(default_factory=dict)
    round_metrics: List[Dict] = field(default_factory=list)
    aggregation_strategy: Optional[AggregationStrategy] = None


class FLCoordinator:
    """Coordinates the federated learning process"""
    
    def __init__(self, state: FLSimulationState, logger: logging.Logger):
        self.state = state
        self.logger = logger
        self.data_loader = None
        self.client_trainer = None
        
    def initialize(self, config: FLConfig):
        """Initialize the FL simulation"""
        self.state.config = config
        self.data_loader = DataLoader(config, self.logger)
        self.client_trainer = ClientTrainer(config, self.logger)
        
        # Set aggregation strategy
        agg_method = AggregationMethod(config.aggregation_method)
        self.state.aggregation_strategy = AggregationFactory.create_aggregator(agg_method)
        
        # Set seeds
        if config.seed is not None:
            tf.keras.utils.set_random_seed(config.seed)
            np.random.seed(config.seed)
        
        # Load and distribute data
        self.logger.info("Loading and distributing data...")
        (client_datasets, client_samples, test_dataset, num_classes) = \
            self.data_loader.load_and_distribute()
        
        self.state.client_train_datasets = client_datasets
        self.state.client_num_samples = client_samples
        self.state.centralized_test_dataset = test_dataset
        self.state.num_classes = num_classes
        
        # Determine eligible clients
        self.state.eligible_client_indices = [
            i for i, n in enumerate(client_samples) if n > 0
        ]
        
        # Create global model
        self.logger.info("Creating global model...")
        dataset_type = DatasetType(config.dataset)
        self.state.global_model = ModelFactory.create_model(
            dataset_type, num_classes, config.seed
        )
        
        # Compile global model
        self.state.global_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        self.state.current_global_weights = self.state.global_model.get_weights()
        
        # Initialize client metadata
        for i in range(config.num_clients):
            self.state.client_metadata[i] = {
                'num_samples': client_samples[i],
                'is_eligible': i in self.state.eligible_client_indices,
                'rounds_participated': 0,
                'total_loss': 0.0,
                'total_accuracy': 0.0
            }
        
        # Set flags
        self.state.simulation_initialized = True
        self.state.model_compiled = True
        self.state.data_loaded = True
        self.state.current_round = 0
        
        # Evaluate initial model
        self._evaluate_and_log(round_num=0)
        
    def run_round(self, selected_clients: Optional[List[int]] = None) -> Dict:
        """Execute one round of federated learning"""
        
        if not self.state.simulation_initialized:
            raise RuntimeError("Simulation not initialized")
        
        if not self.state.eligible_client_indices:
            self.logger.warning("No eligible clients for training")
            return {"message": "No eligible clients", "round": self.state.current_round}
        
        self.state.current_round += 1
        round_num = self.state.current_round
        
        self.logger.info(f"Starting FL Round {round_num}")
        round_start_time = time.time()
        
        # Select clients for this round
        if selected_clients is not None:
            # Validate selected clients
            participating_clients = [
                c for c in selected_clients 
                if c in self.state.eligible_client_indices
            ]
            if len(participating_clients) != len(selected_clients):
                self.logger.warning(
                    f"Some selected clients are not eligible. "
                    f"Using {len(participating_clients)} of {len(selected_clients)}"
                )
        else:
            # Random selection
            num_to_sample = min(
                self.state.config.clients_per_round,
                len(self.state.eligible_client_indices)
            )
            participating_clients = np.random.choice(
                self.state.eligible_client_indices,
                size=num_to_sample,
                replace=False
            ).tolist()
        
        self.logger.info(f"Selected clients: {participating_clients}")
        
        # Train clients and collect updates
        round_results = self._train_clients(participating_clients)
        
        # Aggregate updates
        if round_results['client_weights']:
            new_weights = self.state.aggregation_strategy.aggregate(
                round_results['client_weights'],
                round_results['client_samples']
            )
            
            if new_weights is not None:
                self.state.current_global_weights = new_weights
                self.state.global_model.set_weights(new_weights)
                self.logger.info("Global model updated")
        
        # Calculate round metrics
        round_metrics = self._calculate_round_metrics(round_results)
        
        # Evaluate global model if needed
        if round_num % self.state.config.eval_every == 0:
            test_metrics = self._evaluate_and_log(round_num)
            round_metrics.update(test_metrics)
        
        # Update history
        self._update_history(round_num, round_metrics)
        
        # Store round metadata
        self.state.round_metrics.append({
            'round': round_num,
            'clients': participating_clients,
            'metrics': round_metrics,
            'duration': time.time() - round_start_time
        })
        
        return {
            'round': round_num,
            'selected_clients': participating_clients,
            'metrics': round_metrics,
            'duration': time.time() - round_start_time,
            'client_performance': round_results['client_performance']
        }
    
    def _train_clients(self, client_indices: List[int]) -> Dict:
        """Train selected clients"""
        
        client_weights = []
        client_samples = []
        client_losses = []
        client_accuracies = []
        client_performance = {}
        
        for client_id in client_indices:
            dataset = self.state.client_train_datasets[client_id]
            num_samples = self.state.client_num_samples[client_id]
            
            if num_samples == 0:
                self.logger.warning(f"Client {client_id} has no data")
                continue
            
            self.logger.info(f"Training client {client_id} ({num_samples} samples)")
            
            # Train client
            weights, loss, accuracy = self.client_trainer.train(
                self.state.global_model,
                self.state.current_global_weights,
                dataset,
                num_samples
            )
            
            # Collect results
            client_weights.append(weights)
            client_samples.append(num_samples)
            client_losses.append(loss)
            client_accuracies.append(accuracy)
            
            # Simulate realistic metrics
            model_size = self._estimate_model_size()
            training_time = self._estimate_training_time(num_samples)
            
            client_performance[client_id] = {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'num_samples': num_samples,
                'model_size_bytes': model_size,
                'training_time_ms': training_time
            }
            
            # Update client metadata
            self.state.client_metadata[client_id]['rounds_participated'] += 1
            self.state.client_metadata[client_id]['total_loss'] += loss
            self.state.client_metadata[client_id]['total_accuracy'] += accuracy
        
        return {
            'client_weights': client_weights,
            'client_samples': client_samples,
            'client_losses': client_losses,
            'client_accuracies': client_accuracies,
            'client_performance': client_performance
        }
    
    def _calculate_round_metrics(self, round_results: Dict) -> Dict:
        """Calculate metrics for the round"""
        
        metrics = {
            'avg_client_loss': float('nan'),
            'avg_client_accuracy': float('nan')
        }
        
        if round_results['client_samples'] and sum(round_results['client_samples']) > 0:
            # Weighted average
            total_samples = sum(round_results['client_samples'])
            metrics['avg_client_loss'] = sum(
                l * s for l, s in zip(
                    round_results['client_losses'],
                    round_results['client_samples']
                )
            ) / total_samples
            metrics['avg_client_accuracy'] = sum(
                a * s for a, s in zip(
                    round_results['client_accuracies'],
                    round_results['client_samples']
                )
            ) / total_samples
        elif round_results['client_losses']:
            # Simple average
            metrics['avg_client_loss'] = np.mean(round_results['client_losses'])
            metrics['avg_client_accuracy'] = np.mean(round_results['client_accuracies'])
        
        return metrics
    
    def _evaluate_and_log(self, round_num: int) -> Dict:
        """Evaluate global model on test set"""
        
        self.logger.info(f"Evaluating global model (round {round_num})")
        
        if self.state.centralized_test_dataset and self.state.global_model:
            loss, accuracy = self.state.global_model.evaluate(
                self.state.centralized_test_dataset,
                verbose=0
            )
            
            self.logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
            
            return {
                'global_test_loss': float(loss),
                'global_test_accuracy': float(accuracy)
            }
        
        return {
            'global_test_loss': float('nan'),
            'global_test_accuracy': float('nan')
        }
    
    def _update_history(self, round_num: int, metrics: Dict):
        """Update training history"""
        
        self.state.history_log['round'].append(round_num)
        
        for key in ['avg_client_loss', 'avg_client_accuracy', 
                   'global_test_loss', 'global_test_accuracy']:
            value = metrics.get(key, float('nan'))
            self.state.history_log[key].append(value)
    
    def _estimate_model_size(self) -> int:
        """Estimate model size in bytes"""
        
        if not self.state.current_global_weights:
            return 0
        
        total_params = sum(
            w.size for w in self.state.current_global_weights
        )
        # Assume float32 (4 bytes per parameter)
        return int(total_params * 4)
    
    def _estimate_training_time(self, num_samples: int) -> int:
        """Estimate training time in milliseconds"""
        
        # Base time + time per sample * epochs
        base_time = 100  # ms
        time_per_sample = 0.5  # ms
        
        return int(
            base_time + 
            num_samples * time_per_sample * self.state.config.local_epochs
        )
    
    def reset(self):
        """Reset the simulation state"""
        
        self.state.config = None
        self.state.client_train_datasets = None
        self.state.client_num_samples = None
        self.state.centralized_test_dataset = None
        self.state.num_classes = None
        self.state.global_model = None
        self.state.current_global_weights = None
        self.state.eligible_client_indices = None
        self.state.history_log = collections.defaultdict(list)
        self.state.current_round = 0
        self.state.simulation_initialized = False
        self.state.model_compiled = False
        self.state.data_loaded = False
        self.state.is_training_round_active = False
        self.state.client_metadata = {}
        self.state.round_metrics = []
        self.state.aggregation_strategy = None
        
        # Clear TensorFlow session
        tf.keras.backend.clear_session()