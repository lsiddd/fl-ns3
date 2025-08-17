"""
Tests for configuration classes and enums in the FL API.
"""

import pytest
from config import (
    FLConfig, DatasetType, NonIIDType, QuantitySkewType, 
    FeatureSkewType, OptimizerType, AggregationMethod
)


class TestEnums:
    """Test enum classes"""
    
    def test_dataset_type_enum(self):
        assert DatasetType.MNIST.value == 'mnist'
        assert DatasetType.EMNIST_DIGITS.value == 'emnist_digits'
        assert DatasetType.EMNIST_CHAR.value == 'emnist_char'
        assert DatasetType.CIFAR10.value == 'cifar10'
        
        # Test enum creation from string
        assert DatasetType('mnist') == DatasetType.MNIST
        
        # Test invalid enum value
        with pytest.raises(ValueError):
            DatasetType('invalid_dataset')
    
    def test_non_iid_type_enum(self):
        assert NonIIDType.IID.value == 'iid'
        assert NonIIDType.PATHOLOGICAL.value == 'pathological'
        assert NonIIDType.DIRICHLET.value == 'dirichlet'
        
        assert NonIIDType('iid') == NonIIDType.IID
        
        with pytest.raises(ValueError):
            NonIIDType('invalid_type')
    
    def test_quantity_skew_type_enum(self):
        assert QuantitySkewType.UNIFORM.value == 'uniform'
        assert QuantitySkewType.POWER_LAW.value == 'power_law'
        
        assert QuantitySkewType('uniform') == QuantitySkewType.UNIFORM
        
        with pytest.raises(ValueError):
            QuantitySkewType('invalid_skew')
    
    def test_feature_skew_type_enum(self):
        assert FeatureSkewType.NONE.value == 'none'
        assert FeatureSkewType.NOISE.value == 'noise'
        
        assert FeatureSkewType('none') == FeatureSkewType.NONE
        
        with pytest.raises(ValueError):
            FeatureSkewType('invalid_feature_skew')
    
    def test_optimizer_type_enum(self):
        assert OptimizerType.SGD.value == 'sgd'
        assert OptimizerType.ADAM.value == 'adam'
        
        assert OptimizerType('sgd') == OptimizerType.SGD
        
        with pytest.raises(ValueError):
            OptimizerType('invalid_optimizer')
    
    def test_aggregation_method_enum(self):
        assert AggregationMethod.FEDAVG.value == 'fedavg'
        assert AggregationMethod.FEDPROX.value == 'fedprox'
        assert AggregationMethod.FEDOPT.value == 'fedopt'
        
        assert AggregationMethod('fedavg') == AggregationMethod.FEDAVG
        
        with pytest.raises(ValueError):
            AggregationMethod('invalid_aggregation')


class TestFLConfig:
    """Test FLConfig class"""
    
    def test_default_config(self):
        config = FLConfig()
        
        # Test default values
        assert config.dataset == 'mnist'
        assert config.num_clients == 10
        assert config.non_iid_type == 'iid'
        assert config.non_iid_alpha == 0.5
        assert config.quantity_skew_type == 'uniform'
        assert config.power_law_beta == 2.0
        assert config.feature_skew_type == 'none'
        assert config.noise_std_dev == 0.1
        assert config.clients_per_round == 5
        assert config.num_rounds_api_max == 100
        assert config.local_epochs == 1
        assert config.batch_size == 32
        assert config.client_optimizer == 'sgd'
        assert config.client_lr == 0.01
        assert config.aggregation_method == 'fedavg'
        assert config.eval_every == 1
        assert config.seed == 42
        assert config.port == 5000
    
    def test_custom_config(self):
        config = FLConfig(
            dataset='cifar10',
            num_clients=20,
            non_iid_type='pathological',
            non_iid_alpha=2.0,
            clients_per_round=10,
            batch_size=64,
            client_optimizer='adam',
            client_lr=0.001
        )
        
        assert config.dataset == 'cifar10'
        assert config.num_clients == 20
        assert config.non_iid_type == 'pathological'
        assert config.non_iid_alpha == 2.0
        assert config.clients_per_round == 10
        assert config.batch_size == 64
        assert config.client_optimizer == 'adam'
        assert config.client_lr == 0.001
    
    def test_from_dict(self):
        config_dict = {
            'dataset': 'emnist_digits',
            'num_clients': 15,
            'non_iid_type': 'dirichlet',
            'non_iid_alpha': 0.1,
            'batch_size': 16,
            'client_lr': 0.005,
            'extra_field': 'should_be_ignored'  # Should be filtered out
        }
        
        config = FLConfig.from_dict(config_dict)
        
        assert config.dataset == 'emnist_digits'
        assert config.num_clients == 15
        assert config.non_iid_type == 'dirichlet'
        assert config.non_iid_alpha == 0.1
        assert config.batch_size == 16
        assert config.client_lr == 0.005
        
        # Default values should remain for unspecified fields
        assert config.local_epochs == 1
        assert config.seed == 42
        
        # Extra field should not be present
        assert not hasattr(config, 'extra_field')
    
    def test_validate_valid_config(self):
        config = FLConfig()
        valid, error = config.validate()
        
        assert valid is True
        assert error is None
    
    def test_validate_invalid_dataset(self):
        config = FLConfig(dataset='invalid_dataset')
        valid, error = config.validate()
        
        assert valid is False
        assert 'invalid_dataset' in str(error)
    
    def test_validate_invalid_non_iid_type(self):
        config = FLConfig(non_iid_type='invalid_type')
        valid, error = config.validate()
        
        assert valid is False
        assert 'invalid_type' in str(error)
    
    def test_validate_invalid_quantity_skew_type(self):
        config = FLConfig(quantity_skew_type='invalid_skew')
        valid, error = config.validate()
        
        assert valid is False
        assert 'invalid_skew' in str(error)
    
    def test_validate_invalid_feature_skew_type(self):
        config = FLConfig(feature_skew_type='invalid_feature')
        valid, error = config.validate()
        
        assert valid is False
        assert 'invalid_feature' in str(error)
    
    def test_validate_invalid_optimizer(self):
        config = FLConfig(client_optimizer='invalid_opt')
        valid, error = config.validate()
        
        assert valid is False
        assert 'invalid_opt' in str(error)
    
    def test_validate_invalid_aggregation_method(self):
        config = FLConfig(aggregation_method='invalid_agg')
        valid, error = config.validate()
        
        assert valid is False
        assert 'invalid_agg' in str(error)
    
    def test_validate_invalid_num_clients(self):
        config = FLConfig(num_clients=0)
        valid, error = config.validate()
        
        assert valid is False
        assert 'num_clients must be positive' in error
        
        config = FLConfig(num_clients=-5)
        valid, error = config.validate()
        
        assert valid is False
        assert 'num_clients must be positive' in error
    
    def test_validate_invalid_clients_per_round(self):
        config = FLConfig(num_clients=10, clients_per_round=0)
        valid, error = config.validate()
        
        assert valid is False
        assert 'clients_per_round must be between 1 and num_clients' in error
        
        config = FLConfig(num_clients=10, clients_per_round=15)
        valid, error = config.validate()
        
        assert valid is False
        assert 'clients_per_round must be between 1 and num_clients' in error
    
    def test_validate_invalid_batch_size(self):
        config = FLConfig(batch_size=0)
        valid, error = config.validate()
        
        assert valid is False
        assert 'batch_size must be positive' in error
        
        config = FLConfig(batch_size=-10)
        valid, error = config.validate()
        
        assert valid is False
        assert 'batch_size must be positive' in error
    
    def test_validate_invalid_client_lr(self):
        config = FLConfig(client_lr=0.0)
        valid, error = config.validate()
        
        assert valid is False
        assert 'client_lr must be positive' in error
        
        config = FLConfig(client_lr=-0.01)
        valid, error = config.validate()
        
        assert valid is False
        assert 'client_lr must be positive' in error
    
    def test_validate_boundary_values(self):
        # Test minimum valid values
        config = FLConfig(
            num_clients=1,
            clients_per_round=1,
            batch_size=1,
            client_lr=0.0001
        )
        valid, error = config.validate()
        
        assert valid is True
        assert error is None
        
        # Test clients_per_round equal to num_clients
        config = FLConfig(num_clients=5, clients_per_round=5)
        valid, error = config.validate()
        
        assert valid is True
        assert error is None


class TestConfigIntegration:
    """Integration tests for config usage"""
    
    def test_config_enum_validation_integration(self):
        """Test that config validation works with actual enum instances"""
        
        # Test with valid enum instances
        config = FLConfig(
            dataset=DatasetType.CIFAR10.value,
            non_iid_type=NonIIDType.PATHOLOGICAL.value,
            quantity_skew_type=QuantitySkewType.POWER_LAW.value,
            feature_skew_type=FeatureSkewType.NOISE.value,
            client_optimizer=OptimizerType.ADAM.value,
            aggregation_method=AggregationMethod.FEDAVG.value
        )
        
        valid, error = config.validate()
        assert valid is True
        assert error is None
    
    def test_config_serialization(self):
        """Test config can be properly serialized and deserialized"""
        original_config = FLConfig(
            dataset='emnist_char',
            num_clients=50,
            non_iid_type='dirichlet',
            non_iid_alpha=0.3,
            clients_per_round=20,
            batch_size=128,
            client_lr=0.002
        )
        
        # Convert to dict (simulate serialization)
        config_dict = {
            'dataset': original_config.dataset,
            'num_clients': original_config.num_clients,
            'non_iid_type': original_config.non_iid_type,
            'non_iid_alpha': original_config.non_iid_alpha,
            'clients_per_round': original_config.clients_per_round,
            'batch_size': original_config.batch_size,
            'client_lr': original_config.client_lr
        }
        
        # Recreate config from dict
        restored_config = FLConfig.from_dict(config_dict)
        
        # Verify values match
        assert restored_config.dataset == original_config.dataset
        assert restored_config.num_clients == original_config.num_clients
        assert restored_config.non_iid_type == original_config.non_iid_type
        assert restored_config.non_iid_alpha == original_config.non_iid_alpha
        assert restored_config.clients_per_round == original_config.clients_per_round
        assert restored_config.batch_size == original_config.batch_size
        assert restored_config.client_lr == original_config.client_lr
        
        # Verify validation still works
        valid, error = restored_config.validate()
        assert valid is True
        assert error is None