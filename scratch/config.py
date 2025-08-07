"""
Configuration classes and enums for Federated Learning simulation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


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