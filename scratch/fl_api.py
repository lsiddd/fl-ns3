# Filename: scratch/sim/fl_api.py
"""
Federated Learning API Server

A production-ready FL simulation server with proper separation of concerns,
extensible architecture, and realistic federated learning implementation.
"""

import collections
import time
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, request, jsonify
import threading
import logging
import socket
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
from abc import ABC, abstractmethod

# ============================================================================
# Configuration and Setup
# ============================================================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# Enums and Constants
# ============================================================================

class DatasetType(Enum):
    """Supported dataset types"""
    MNIST = 'mnist'
    EMNIST_DIGITS = 'emnist_digits'
    EMNIST_CHAR = 'emnist_char'
    CIFAR10 = 'cifar10'

class NonIIDType(Enum):
    """Data distribution strategies"""
    IID = 'iid'
    PATHOLOGICAL = 'pathological'
    DIRICHLET = 'dirichlet'

class QuantitySkewType(Enum):
    """Quantity distribution strategies"""
    UNIFORM = 'uniform'
    POWER_LAW = 'power_law'

class FeatureSkewType(Enum):
    """Feature distribution strategies"""
    NONE = 'none'
    NOISE = 'noise'

class OptimizerType(Enum):
    """Supported optimizers"""
    SGD = 'sgd'
    ADAM = 'adam'

class AggregationMethod(Enum):
    """Federated aggregation methods"""
    FEDAVG = 'fedavg'
    FEDPROX = 'fedprox'  # Placeholder for extension
    FEDOPT = 'fedopt'    # Placeholder for extension

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class FLConfig:
    """Federated Learning configuration"""
    # Dataset parameters
    dataset: str = 'mnist'
    num_clients: int = 10
    non_iid_type: str = 'iid'
    non_iid_alpha: float = 0.5
    quantity_skew_type: str = 'uniform'
    power_law_beta: float = 2.0
    feature_skew_type: str = 'none'
    noise_std_dev: float = 0.1
    
    # Training parameters
    clients_per_round: int = 5
    num_rounds_api_max: int = 100
    local_epochs: int = 1
    batch_size: int = 32
    client_optimizer: str = 'sgd'
    client_lr: float = 0.01
    aggregation_method: str = 'fedavg'
    
    # Evaluation parameters
    eval_every: int = 1
    seed: int = 42
    
    # Network parameters
    port: int = 5000
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FLConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration"""
        try:
            # Validate enums
            DatasetType(self.dataset)
            NonIIDType(self.non_iid_type)
            QuantitySkewType(self.quantity_skew_type)
            FeatureSkewType(self.feature_skew_type)
            OptimizerType(self.client_optimizer)
            AggregationMethod(self.aggregation_method)
            
            # Validate numeric constraints
            if self.num_clients <= 0:
                return False, "num_clients must be positive"
            if self.clients_per_round <= 0 or self.clients_per_round > self.num_clients:
                return False, "clients_per_round must be between 1 and num_clients"
            if self.batch_size <= 0:
                return False, "batch_size must be positive"
            if self.client_lr <= 0:
                return False, "client_lr must be positive"
            
            return True, None
        except ValueError as e:
            return False, str(e)

# ============================================================================
# Data Management Classes
# ============================================================================

class DataPartitioner:
    """Handles data partitioning strategies for federated learning"""
    
    def __init__(self, config: FLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def partition_data(self, x_data: np.ndarray, y_data: np.ndarray, 
                       num_classes: int) -> List[List[int]]:
        """Partition data according to configuration"""
        non_iid_type = NonIIDType(self.config.non_iid_type)
        
        if non_iid_type == NonIIDType.IID:
            return self._partition_iid(x_data, y_data)
        elif non_iid_type == NonIIDType.PATHOLOGICAL:
            return self._partition_pathological(x_data, y_data, num_classes)
        elif non_iid_type == NonIIDType.DIRICHLET:
            return self._partition_dirichlet(x_data, y_data, num_classes)
        else:
            raise ValueError(f"Unknown non-IID type: {self.config.non_iid_type}")
    
    def _partition_iid(self, x_data: np.ndarray, y_data: np.ndarray) -> List[List[int]]:
        """IID data partitioning"""
        self.logger.info("Applying IID data partitioning")
        all_indices = np.arange(len(y_data))
        np.random.shuffle(all_indices)
        idx_splits = np.array_split(all_indices, self.config.num_clients)
        return [idx.tolist() for idx in idx_splits]
    
    def _partition_pathological(self, x_data: np.ndarray, y_data: np.ndarray, 
                                num_classes: int) -> List[List[int]]:
        """Pathological non-IID partitioning"""
        num_classes_per_client = int(max(1, self.config.non_iid_alpha))
        self.logger.info(f"Applying Pathological non-IID: ~{num_classes_per_client} classes per client")
        
        indices_by_label = [np.where(y_data == i)[0] for i in range(num_classes)]
        for idx_list in indices_by_label:
            np.random.shuffle(idx_list)
        
        # Create shards
        all_shards = []
        for label_idx, indices_list in enumerate(indices_by_label):
            num_shards = max(2, self.config.num_clients // num_classes)
            if len(indices_list) > 0:
                shards = np.array_split(indices_list, num_shards)
                for shard in shards:
                    if len(shard) > 0:
                        all_shards.append((label_idx, shard.tolist()))
        
        np.random.shuffle(all_shards)
        
        # Distribute shards to clients
        client_data_indices = [[] for _ in range(self.config.num_clients)]
        client_labels = [set() for _ in range(self.config.num_clients)]
        
        shard_idx = 0
        rounds = 0
        max_rounds = 3  # Prevent infinite loop
        
        while shard_idx < len(all_shards) and rounds < max_rounds:
            rounds += 1
            for client_id in range(self.config.num_clients):
                if shard_idx >= len(all_shards):
                    break
                    
                label, indices = all_shards[shard_idx]
                
                # Assign if client has room for new class or already has this class
                if (len(client_labels[client_id]) < num_classes_per_client or 
                    label in client_labels[client_id]):
                    client_data_indices[client_id].extend(indices)
                    client_labels[client_id].add(label)
                    shard_idx += 1
        
        # Clean up duplicates
        for i in range(self.config.num_clients):
            client_data_indices[i] = list(np.unique(client_data_indices[i]))
            
        return client_data_indices
    
    def _partition_dirichlet(self, x_data: np.ndarray, y_data: np.ndarray, 
                             num_classes: int) -> List[List[int]]:
        """Dirichlet-based non-IID partitioning"""
        self.logger.info(f"Applying Dirichlet non-IID with alpha={self.config.non_iid_alpha}")
        
        label_proportions = np.random.dirichlet(
            [self.config.non_iid_alpha] * num_classes, 
            self.config.num_clients
        )
        
        indices_by_label = [np.where(y_data == i)[0] for i in range(num_classes)]
        for idx_list in indices_by_label:
            np.random.shuffle(idx_list)
        
        client_data_indices = [[] for _ in range(self.config.num_clients)]
        ptr_by_label = [0] * num_classes
        
        for client_idx in range(self.config.num_clients):
            samples_per_client = len(y_data) // self.config.num_clients
            
            for label_k in range(num_classes):
                if len(indices_by_label[label_k]) == 0:
                    continue
                    
                num_to_take = int(label_proportions[client_idx, label_k] * samples_per_client)
                start_ptr = ptr_by_label[label_k]
                available = len(indices_by_label[label_k]) - start_ptr
                
                actual_take = min(num_to_take, available)
                if actual_take > 0:
                    end_ptr = start_ptr + actual_take
                    client_data_indices[client_idx].extend(
                        indices_by_label[label_k][start_ptr:end_ptr]
                    )
                    ptr_by_label[label_k] = end_ptr
            
            np.random.shuffle(client_data_indices[client_idx])
            
        return client_data_indices
    
    def apply_quantity_skew(self, client_indices: List[List[int]]) -> List[List[int]]:
        """Apply quantity skew to client data"""
        skew_type = QuantitySkewType(self.config.quantity_skew_type)
        
        if skew_type == QuantitySkewType.UNIFORM:
            return client_indices
        elif skew_type == QuantitySkewType.POWER_LAW:
            return self._apply_power_law_skew(client_indices)
        else:
            raise ValueError(f"Unknown quantity skew type: {self.config.quantity_skew_type}")
    
    def _apply_power_law_skew(self, client_indices: List[List[int]]) -> List[List[int]]:
        """Apply power law distribution to data quantities"""
        self.logger.info(f"Applying Power Law Quantity Skew with beta={self.config.power_law_beta}")
        
        min_samples = max(1, self.config.batch_size // 4)
        raw_samples = np.random.pareto(self.config.power_law_beta, self.config.num_clients) + 1e-6
        proportions = raw_samples / np.sum(raw_samples)
        
        total_samples = sum(len(idx_list) for idx_list in client_indices)
        target_samples = (proportions * total_samples).astype(int)
        target_samples = np.maximum(target_samples, min_samples)
        
        # Adjust to not exceed total
        if np.sum(target_samples) > total_samples:
            scale = total_samples / np.sum(target_samples)
            target_samples = (target_samples * scale).astype(int)
            target_samples = np.maximum(target_samples, min_samples)
        
        new_indices = []
        for i, indices in enumerate(client_indices):
            np.random.shuffle(indices)
            target = min(target_samples[i], len(indices))
            new_indices.append(indices[:target])
            
        return new_indices

# ============================================================================
# Model Management Classes
# ============================================================================

class ModelFactory:
    """Factory for creating FL models"""
    
    @staticmethod
    def create_model(dataset_type: DatasetType, num_classes: int, 
                    seed: Optional[int] = None) -> tf.keras.Model:
        """Create a model based on dataset type"""
        
        if dataset_type in [DatasetType.MNIST, DatasetType.EMNIST_DIGITS]:
            input_shape = (28, 28, 1)
            if num_classes is None:
                num_classes = 10
        elif dataset_type == DatasetType.EMNIST_CHAR:
            input_shape = (28, 28, 1)
            if num_classes is None:
                num_classes = 62
        elif dataset_type == DatasetType.CIFAR10:
            input_shape = (32, 32, 3)
            if num_classes is None:
                num_classes = 10
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Build model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation='relu',
                input_shape=input_shape,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        ])
        
        # Add extra conv layer for CIFAR10
        if dataset_type == DatasetType.CIFAR10:
            model.add(tf.keras.layers.Conv2D(
                64, (3, 3), activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
            ))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
        # Add dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            128, activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        ))
        model.add(tf.keras.layers.Dense(
            num_classes, activation='softmax',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        ))
        
        return model

# ============================================================================
# Aggregation Strategies
# ============================================================================

class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies"""
    
    @abstractmethod
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                 client_samples: List[int]) -> List[np.ndarray]:
        """Aggregate client weights"""
        pass

class FedAvgAggregation(AggregationStrategy):
    """Federated Averaging aggregation"""
    
    def aggregate(self, client_weights: List[List[np.ndarray]], 
                 client_samples: List[int]) -> List[np.ndarray]:
        """Weighted average of client weights"""
        if not client_weights:
            return None
            
        total_samples = sum(client_samples)
        if total_samples == 0:
            return client_weights[0] if client_weights else None
        
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        for i, weights in enumerate(client_weights):
            if client_samples[i] == 0:
                continue
            weight_factor = client_samples[i] / total_samples
            for j, layer_weights in enumerate(weights):
                avg_weights[j] += weight_factor * layer_weights
                
        return avg_weights

class AggregationFactory:
    """Factory for creating aggregation strategies"""
    
    @staticmethod
    def create_aggregator(method: AggregationMethod) -> AggregationStrategy:
        """Create an aggregation strategy"""
        if method == AggregationMethod.FEDAVG:
            return FedAvgAggregation()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

# ============================================================================
# Client Trainer
# ============================================================================

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

# ============================================================================
# Federated Learning Simulation State
# ============================================================================

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

# ============================================================================
# Data Loader
# ============================================================================

class DataLoader:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, config: FLConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.partitioner = DataPartitioner(config, logger)
        
    def load_and_distribute(self) -> Tuple[List[tf.data.Dataset], List[int], 
                                          tf.data.Dataset, int]:
        """Load dataset and distribute to clients"""
        
        # Set random seeds
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Load dataset
        dataset_type = DatasetType(self.config.dataset)
        x_train, y_train, x_test, y_test, num_classes = self._load_dataset(dataset_type)
        
        # Partition data
        client_indices = self.partitioner.partition_data(x_train, y_train, num_classes)
        
        # Apply quantity skew
        client_indices = self.partitioner.apply_quantity_skew(client_indices)
        
        # Create client datasets
        client_datasets = []
        client_num_samples = []
        
        for i, indices in enumerate(client_indices):
            if len(indices) == 0:
                # Empty dataset
                empty_ds = self._create_empty_dataset()
                client_datasets.append(empty_ds)
                client_num_samples.append(0)
            else:
                client_x = x_train[indices]
                client_y = y_train[indices]
                
                # Apply feature skew if configured
                client_x = self._apply_feature_skew(client_x, i)
                
                # Create TF dataset
                client_ds = self._create_tf_dataset(client_x, client_y)
                client_datasets.append(client_ds)
                client_num_samples.append(len(indices))
        
        # Create test dataset
        test_dataset = self._create_test_dataset(x_test, y_test)
        
        return client_datasets, client_num_samples, test_dataset, num_classes
    
    def _load_dataset(self, dataset_type: DatasetType) -> Tuple[np.ndarray, np.ndarray, 
                                                                np.ndarray, np.ndarray, int]:
        """Load the specified dataset"""
        
        tfds_mapping = {
            DatasetType.MNIST: 'mnist',
            DatasetType.EMNIST_DIGITS: 'emnist/digits',
            DatasetType.EMNIST_CHAR: 'emnist/byclass',
            DatasetType.CIFAR10: 'cifar10'
        }
        
        tfds_name = tfds_mapping[dataset_type]
        self.logger.info(f"Loading dataset '{tfds_name}'...")
        
        # Load dataset
        train_ds, ds_info = tfds.load(
            tfds_name, split='train', 
            as_supervised=False, with_info=True, 
            shuffle_files=True
        )
        test_ds = tfds.load(tfds_name, split='test', as_supervised=False)
        
        num_classes = ds_info.features['label'].num_classes
        
        # Convert to numpy
        self.logger.info("Converting dataset to NumPy arrays...")
        train_samples = list(tfds.as_numpy(train_ds))
        test_samples = list(tfds.as_numpy(test_ds))
        
        x_train = np.array([s['image'] for s in train_samples])
        y_train = np.array([s['label'] for s in train_samples])
        x_test = np.array([s['image'] for s in test_samples])
        y_test = np.array([s['label'] for s in test_samples])
        
        # Flatten labels if needed
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.flatten()
        if y_test.ndim > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
            
        return x_train, y_train, x_test, y_test, num_classes
    
    def _apply_feature_skew(self, x_data: np.ndarray, client_id: int) -> np.ndarray:
        """Apply feature skew to client data"""
        skew_type = FeatureSkewType(self.config.feature_skew_type)
        
        if skew_type == FeatureSkewType.NONE:
            return x_data.astype('float32')
        elif skew_type == FeatureSkewType.NOISE:
            x_processed = x_data.astype('float32')
            noise_std = self.config.noise_std_dev * 255.0
            noise = np.random.normal(0, noise_std, x_processed.shape)
            x_processed = np.clip(x_processed + noise, 0.0, 255.0)
            return x_processed
        else:
            return x_data.astype('float32')
    
    def _create_tf_dataset(self, x_data: np.ndarray, y_data: np.ndarray) -> tf.data.Dataset:
        """Create TensorFlow dataset for client"""
        
        def preprocess(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            if len(x.shape) == 2:
                x = tf.expand_dims(x, axis=-1)
            return x, y
        
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        if len(y_data) > 0:
            dataset = dataset.shuffle(
                buffer_size=len(y_data),
                seed=self.config.seed,
                reshuffle_each_iteration=True
            )
            dataset = dataset.repeat(self.config.local_epochs)
        
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _create_test_dataset(self, x_data: np.ndarray, y_data: np.ndarray) -> tf.data.Dataset:
        """Create test dataset"""
        
        def preprocess(x, y):
            x = tf.cast(x, tf.float32) / 255.0
            if len(x.shape) == 2:
                x = tf.expand_dims(x, axis=-1)
            return x, y
        
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.batch_size * 2)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _create_empty_dataset(self) -> tf.data.Dataset:
        """Create an empty dataset"""
        empty_x = np.array([], dtype=np.float32).reshape(0, 28, 28, 1)
        empty_y = np.array([], dtype=np.int32)
        return self._create_tf_dataset(empty_x, empty_y)

# ============================================================================
# FL Coordinator
# ============================================================================

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

# ============================================================================
# Flask API Server
# ============================================================================

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

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Federated Learning API Server")
    parser.add_argument('--port', type=int, default=5000, 
                       help='Starting port for the server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args, _ = parser.parse_known_args()
    
    # Create and run server
    server = FLAPIServer()
    server.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()