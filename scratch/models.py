"""
Model factory and aggregation strategies for Federated Learning simulation.
"""

import numpy as np
import tensorflow as tf
from typing import List, Optional
from abc import ABC, abstractmethod

from config import DatasetType, AggregationMethod


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