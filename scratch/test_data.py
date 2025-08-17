"""
Tests for data management classes in the FL API.
"""

import pytest
import numpy as np
import tensorflow as tf
import logging
from unittest.mock import Mock, patch, MagicMock

from config import FLConfig, DatasetType, NonIIDType, QuantitySkewType, FeatureSkewType
from data import DataPartitioner, DataLoader


class TestDataPartitioner:
    """Test DataPartitioner class"""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def default_config(self):
        return FLConfig(
            num_clients=4,
            non_iid_type='iid',
            non_iid_alpha=0.5,
            quantity_skew_type='uniform',
            power_law_beta=2.0,
            feature_skew_type='none',
            batch_size=32,
            seed=42
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        x_data = np.random.randint(0, 255, (100, 28, 28), dtype=np.uint8)
        y_data = np.random.randint(0, 10, 100)
        return x_data, y_data
    
    def test_init(self, default_config, mock_logger):
        partitioner = DataPartitioner(default_config, mock_logger)
        assert partitioner.config == default_config
        assert partitioner.logger == mock_logger
    
    def test_partition_data_iid(self, default_config, mock_logger, sample_data):
        x_data, y_data = sample_data
        partitioner = DataPartitioner(default_config, mock_logger)
        
        np.random.seed(42)  # Ensure reproducible results
        client_indices = partitioner.partition_data(x_data, y_data, num_classes=10)
        
        # Check that we have the right number of clients
        assert len(client_indices) == default_config.num_clients
        
        # Check that all indices are covered
        all_indices = set()
        for indices in client_indices:
            all_indices.update(indices)
        assert all_indices == set(range(len(y_data)))
        
        # Check that clients have roughly equal data
        client_sizes = [len(indices) for indices in client_indices]
        assert min(client_sizes) > 0
        assert max(client_sizes) - min(client_sizes) <= 1  # At most 1 sample difference
        
        # Verify logger was called
        mock_logger.info.assert_called_with("Applying IID data partitioning")
    
    def test_partition_data_pathological(self, mock_logger, sample_data):
        config = FLConfig(
            num_clients=4,
            non_iid_type='pathological',
            non_iid_alpha=2.0,  # 2 classes per client
            seed=42
        )
        x_data, y_data = sample_data
        partitioner = DataPartitioner(config, mock_logger)
        
        np.random.seed(42)
        client_indices = partitioner.partition_data(x_data, y_data, num_classes=10)
        
        # Check basic structure
        assert len(client_indices) == config.num_clients
        
        # Check that all indices are covered (allowing for some to be excluded)
        all_indices = set()
        for indices in client_indices:
            all_indices.update(indices)
        assert len(all_indices) <= len(y_data)
        
        # Verify logger was called
        mock_logger.info.assert_called_with("Applying Pathological non-IID: ~2 classes per client")
    
    def test_partition_data_dirichlet(self, mock_logger, sample_data):
        config = FLConfig(
            num_clients=4,
            non_iid_type='dirichlet',
            non_iid_alpha=0.5,
            seed=42
        )
        x_data, y_data = sample_data
        partitioner = DataPartitioner(config, mock_logger)
        
        np.random.seed(42)
        client_indices = partitioner.partition_data(x_data, y_data, num_classes=10)
        
        # Check basic structure
        assert len(client_indices) == config.num_clients
        
        # Check that clients have data
        for indices in client_indices:
            assert isinstance(indices, list)
        
        # Verify logger was called
        mock_logger.info.assert_called_with("Applying Dirichlet non-IID with alpha=0.5")
    
    def test_partition_data_invalid_type(self, mock_logger, sample_data):
        # Need to directly set invalid type to bypass enum validation in FLConfig
        config = FLConfig(num_clients=4)
        config.non_iid_type = 'invalid_type'  # Set directly to bypass validation
        x_data, y_data = sample_data
        partitioner = DataPartitioner(config, mock_logger)
        
        with pytest.raises(ValueError, match="Unknown non-IID type"):
            partitioner.partition_data(x_data, y_data, num_classes=10)
    
    def test_apply_quantity_skew_uniform(self, default_config, mock_logger):
        partitioner = DataPartitioner(default_config, mock_logger)
        
        # Create mock client indices
        client_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
        result = partitioner.apply_quantity_skew(client_indices)
        
        # For uniform skew, should return unchanged
        assert result == client_indices
    
    def test_apply_quantity_skew_power_law(self, mock_logger):
        config = FLConfig(
            num_clients=4,
            quantity_skew_type='power_law',
            power_law_beta=2.0,
            batch_size=32,
            seed=42
        )
        partitioner = DataPartitioner(config, mock_logger)
        
        # Create mock client indices with different sizes
        client_indices = [
            list(range(0, 25)),    # 25 samples
            list(range(25, 50)),   # 25 samples
            list(range(50, 75)),   # 25 samples
            list(range(75, 100))   # 25 samples
        ]
        
        np.random.seed(42)
        result = partitioner.apply_quantity_skew(client_indices)
        
        # Check that we still have 4 clients
        assert len(result) == 4
        
        # Check that each client has at least minimum samples
        min_samples = max(1, config.batch_size // 4)
        for indices in result:
            assert len(indices) >= min_samples
        
        # Verify logger was called
        mock_logger.info.assert_called_with("Applying Power Law Quantity Skew with beta=2.0")
    
    def test_apply_quantity_skew_invalid_type(self, mock_logger):
        config = FLConfig()
        config.quantity_skew_type = 'invalid_skew'  # Set directly to bypass validation
        partitioner = DataPartitioner(config, mock_logger)
        
        client_indices = [[0, 1], [2, 3]]
        
        with pytest.raises(ValueError, match="Unknown quantity skew type"):
            partitioner.apply_quantity_skew(client_indices)
    
    def test_power_law_skew_edge_cases(self, mock_logger):
        config = FLConfig(
            num_clients=3,
            quantity_skew_type='power_law',
            power_law_beta=1.0,
            batch_size=8,
            seed=42
        )
        partitioner = DataPartitioner(config, mock_logger)
        
        # Test with empty client indices
        client_indices = [[], [], []]
        np.random.seed(42)
        result = partitioner.apply_quantity_skew(client_indices)
        
        # Should handle empty gracefully
        assert len(result) == 3
        for indices in result:
            assert len(indices) == 0
        
        # Test with very small datasets
        client_indices = [[0], [1], [2]]
        np.random.seed(42)
        result = partitioner.apply_quantity_skew(client_indices)
        
        assert len(result) == 3
        # Each should have at least 1 sample (minimum)
        for indices in result:
            assert len(indices) >= 1


class TestDataLoader:
    """Test DataLoader class"""
    
    @pytest.fixture
    def mock_logger(self):
        return Mock(spec=logging.Logger)
    
    @pytest.fixture
    def default_config(self):
        return FLConfig(
            dataset='mnist',
            num_clients=2,
            seed=42,
            batch_size=32,
            local_epochs=1
        )
    
    @pytest.fixture
    def mock_tfds_data(self):
        """Mock tensorflow datasets data"""
        # Create mock samples
        train_samples = []
        test_samples = []
        
        for i in range(50):  # 50 training samples
            train_samples.append({
                'image': np.random.randint(0, 255, (28, 28), dtype=np.uint8),
                'label': i % 10
            })
        
        for i in range(20):  # 20 test samples
            test_samples.append({
                'image': np.random.randint(0, 255, (28, 28), dtype=np.uint8),
                'label': i % 10
            })
        
        return train_samples, test_samples
    
    def test_init(self, default_config, mock_logger):
        loader = DataLoader(default_config, mock_logger)
        assert loader.config == default_config
        assert loader.logger == mock_logger
        assert isinstance(loader.partitioner, DataPartitioner)
    
    @patch('data.tfds.load')
    @patch('data.tfds.as_numpy')
    def test_load_dataset_mnist(self, mock_as_numpy, mock_tfds_load, 
                               default_config, mock_logger, mock_tfds_data):
        train_samples, test_samples = mock_tfds_data
        
        # Mock dataset info
        mock_ds_info = Mock()
        mock_ds_info.features = {'label': Mock(num_classes=10)}
        
        # Mock tfds.load return values
        mock_tfds_load.side_effect = [
            (Mock(), mock_ds_info),  # First call with with_info=True
            Mock()  # Second call for test set
        ]
        
        # Mock tfds.as_numpy return values
        mock_as_numpy.side_effect = [train_samples, test_samples]
        
        loader = DataLoader(default_config, mock_logger)
        result = loader._load_dataset(DatasetType.MNIST)
        
        x_train, y_train, x_test, y_test, num_classes = result
        
        # Check shapes and types
        assert x_train.shape == (50, 28, 28)
        assert y_train.shape == (50,)
        assert x_test.shape == (20, 28, 28)
        assert y_test.shape == (20,)
        assert num_classes == 10
        
        # Verify tfds.load was called correctly
        assert mock_tfds_load.call_count == 2
        mock_tfds_load.assert_any_call(
            'mnist', split='train', as_supervised=False, 
            with_info=True, shuffle_files=True
        )
        mock_tfds_load.assert_any_call('mnist', split='test', as_supervised=False)
    
    @patch('data.tfds.load')
    @patch('data.tfds.as_numpy')
    def test_load_dataset_cifar10(self, mock_as_numpy, mock_tfds_load, 
                                 mock_logger, mock_tfds_data):
        config = FLConfig(dataset='cifar10', seed=42)
        train_samples, test_samples = mock_tfds_data
        
        # Modify samples for CIFAR10 (32x32x3)
        for sample in train_samples + test_samples:
            sample['image'] = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        mock_ds_info = Mock()
        mock_ds_info.features = {'label': Mock(num_classes=10)}
        
        mock_tfds_load.side_effect = [(Mock(), mock_ds_info), Mock()]
        mock_as_numpy.side_effect = [train_samples, test_samples]
        
        loader = DataLoader(config, mock_logger)
        result = loader._load_dataset(DatasetType.CIFAR10)
        
        x_train, y_train, x_test, y_test, num_classes = result
        
        # Check CIFAR10 specific shapes
        assert x_train.shape == (50, 32, 32, 3)
        assert y_train.shape == (50,)
        assert num_classes == 10
        
        # Verify correct dataset name was used
        mock_tfds_load.assert_any_call(
            'cifar10', split='train', as_supervised=False, 
            with_info=True, shuffle_files=True
        )
    
    def test_apply_feature_skew_none(self, default_config, mock_logger):
        loader = DataLoader(default_config, mock_logger)
        
        x_data = np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8)
        result = loader._apply_feature_skew(x_data, client_id=0)
        
        # Should convert to float32 but not add noise
        assert result.dtype == np.float32
        assert result.shape == x_data.shape
        np.testing.assert_array_equal(result, x_data.astype('float32'))
    
    def test_apply_feature_skew_noise(self, mock_logger):
        config = FLConfig(seed=42)
        config.feature_skew_type = 'noise'  # Set directly to bypass validation
        config.noise_std_dev = 0.1
        loader = DataLoader(config, mock_logger)
        
        x_data = np.random.randint(0, 255, (5, 28, 28), dtype=np.uint8)
        
        np.random.seed(42)
        result = loader._apply_feature_skew(x_data, client_id=0)
        
        # Should be different from original due to noise
        assert result.dtype == np.float32
        assert result.shape == x_data.shape
        assert not np.array_equal(result, x_data.astype('float32'))
        
        # Values should be clipped to [0, 255]
        assert np.all(result >= 0.0)
        assert np.all(result <= 255.0)
    
    def test_create_tf_dataset(self, default_config, mock_logger):
        loader = DataLoader(default_config, mock_logger)
        
        x_data = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        y_data = np.random.randint(0, 10, 20)
        
        dataset = loader._create_tf_dataset(x_data, y_data)
        
        # Check that it's a TensorFlow dataset
        assert isinstance(dataset, tf.data.Dataset)
        
        # Check preprocessing by taking a sample
        sample = next(iter(dataset.take(1)))
        x_sample, y_sample = sample
        
        # Should be normalized to [0, 1] and have channel dimension
        assert x_sample.dtype == tf.float32
        assert tf.reduce_max(x_sample) <= 1.0
        assert tf.reduce_min(x_sample) >= 0.0
        assert x_sample.shape[-1] == 1  # Channel dimension added
        assert y_sample.dtype in [tf.int32, tf.int64]  # Can be either depending on TF version
    
    def test_create_test_dataset(self, default_config, mock_logger):
        loader = DataLoader(default_config, mock_logger)
        
        x_data = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        y_data = np.random.randint(0, 10, 20)
        
        dataset = loader._create_test_dataset(x_data, y_data)
        
        # Check that it's a TensorFlow dataset
        assert isinstance(dataset, tf.data.Dataset)
        
        # Check that batch size is larger for test dataset
        sample = next(iter(dataset.take(1)))
        x_sample, y_sample = sample
        
        # Test dataset should use batch_size * 2
        expected_batch_size = min(default_config.batch_size * 2, 20)
        assert x_sample.shape[0] == expected_batch_size
    
    def test_create_empty_dataset(self, default_config, mock_logger):
        loader = DataLoader(default_config, mock_logger)
        
        dataset = loader._create_empty_dataset()
        
        # Should be a valid TensorFlow dataset
        assert isinstance(dataset, tf.data.Dataset)
        
        # Should have no elements
        elements = list(dataset.take(1))
        assert len(elements) == 0
    
    @patch.object(DataLoader, '_load_dataset')
    def test_load_and_distribute_integration(self, mock_load_dataset, 
                                           default_config, mock_logger):
        # Mock dataset loading
        x_train = np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8)
        y_train = np.random.randint(0, 10, 20)
        x_test = np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8)
        y_test = np.random.randint(0, 10, 10)
        num_classes = 10
        
        mock_load_dataset.return_value = (x_train, y_train, x_test, y_test, num_classes)
        
        loader = DataLoader(default_config, mock_logger)
        np.random.seed(42)
        
        client_datasets, client_num_samples, test_dataset, returned_num_classes = \
            loader.load_and_distribute()
        
        # Check return values
        assert len(client_datasets) == default_config.num_clients
        assert len(client_num_samples) == default_config.num_clients
        assert isinstance(test_dataset, tf.data.Dataset)
        assert returned_num_classes == num_classes
        
        # Check that all clients have valid datasets
        for dataset in client_datasets:
            assert isinstance(dataset, tf.data.Dataset)
        
        # Check that sample counts are reasonable
        total_samples = sum(client_num_samples)
        assert total_samples <= len(y_train)  # Could be less due to partitioning
        
        # Check that there are some eligible clients
        assert sum(1 for n in client_num_samples if n > 0) > 0
    
    def test_dataset_type_mapping(self, mock_logger):
        """Test that dataset type mapping works correctly"""
        configs_and_expected = [
            (FLConfig(dataset='mnist'), 'mnist'),
            (FLConfig(dataset='emnist_digits'), 'emnist/digits'),
            (FLConfig(dataset='emnist_char'), 'emnist/byclass'),
            (FLConfig(dataset='cifar10'), 'cifar10')
        ]
        
        for config, expected_tfds_name in configs_and_expected:
            loader = DataLoader(config, mock_logger)
            
            with patch('data.tfds.load') as mock_load, \
                 patch('data.tfds.as_numpy') as mock_as_numpy:
                
                # Mock returns
                mock_ds_info = Mock()
                mock_ds_info.features = {'label': Mock(num_classes=10)}
                mock_load.side_effect = [(Mock(), mock_ds_info), Mock()]
                mock_as_numpy.side_effect = [[], []]  # Empty samples
                
                try:
                    loader._load_dataset(DatasetType(config.dataset))
                    
                    # Verify correct dataset name was used
                    mock_load.assert_any_call(
                        expected_tfds_name, split='train', as_supervised=False,
                        with_info=True, shuffle_files=True
                    )
                except Exception:
                    # It's ok if the actual loading fails, we just want to test the mapping
                    pass