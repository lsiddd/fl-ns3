"""
Tests for model factory and aggregation strategies in the FL API.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch

from config import DatasetType, AggregationMethod
from models import ModelFactory, AggregationStrategy, FedAvgAggregation, AggregationFactory


class TestModelFactory:
    """Test ModelFactory class"""
    
    def test_create_model_mnist(self):
        model = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        
        # Check that it's a Keras model
        assert isinstance(model, tf.keras.Model)
        
        # Check input shape for MNIST
        assert model.input_shape == (None, 28, 28, 1)
        
        # Check output shape (10 classes)
        assert model.output_shape == (None, 10)
        
        # Check that model has the expected layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        expected_layers = ['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dense']
        assert layer_types == expected_layers
        
        # Check activation functions
        assert model.layers[0].activation.__name__ == 'relu'  # First Conv2D
        assert model.layers[3].activation.__name__ == 'relu'  # First Dense
        assert model.layers[4].activation.__name__ == 'softmax'  # Output Dense
    
    def test_create_model_emnist_digits(self):
        model = ModelFactory.create_model(DatasetType.EMNIST_DIGITS, num_classes=10, seed=42)
        
        # Should be same as MNIST
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 10)
    
    def test_create_model_emnist_char(self):
        model = ModelFactory.create_model(DatasetType.EMNIST_CHAR, num_classes=62, seed=42)
        
        # Should have 62 classes for EMNIST characters
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 28, 28, 1)
        assert model.output_shape == (None, 62)
    
    def test_create_model_cifar10(self):
        model = ModelFactory.create_model(DatasetType.CIFAR10, num_classes=10, seed=42)
        
        # Check input shape for CIFAR10
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 32, 32, 3)
        assert model.output_shape == (None, 10)
        
        # CIFAR10 should have extra conv layer
        layer_types = [type(layer).__name__ for layer in model.layers]
        expected_layers = ['Conv2D', 'MaxPooling2D', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dense']
        assert layer_types == expected_layers
    
    def test_create_model_unknown_dataset(self):
        with pytest.raises(ValueError, match="Unknown dataset type"):
            # Create a mock dataset type that doesn't exist
            class UnknownDataset:
                pass
            
            ModelFactory.create_model(UnknownDataset(), num_classes=10)
    
    def test_create_model_with_seed(self):
        """Test that models created with same seed are identical"""
        model1 = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        model2 = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        
        # Compare initial weights
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        
        assert len(weights1) == len(weights2)
        for w1, w2 in zip(weights1, weights2):
            np.testing.assert_array_equal(w1, w2)
    
    def test_create_model_different_seeds(self):
        """Test that models created with different seeds are different"""
        model1 = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        model2 = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=123)
        
        # Compare initial weights
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        
        # At least some weights should be different
        weights_different = False
        for w1, w2 in zip(weights1, weights2):
            if not np.array_equal(w1, w2):
                weights_different = True
                break
        
        assert weights_different, "Models with different seeds should have different weights"
    
    def test_model_architecture_details(self):
        """Test specific architecture details"""
        model = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        
        # Check first Conv2D layer
        conv1 = model.layers[0]
        assert conv1.filters == 32
        assert conv1.kernel_size == (3, 3)
        
        # Check MaxPooling2D layer
        pool1 = model.layers[1]
        assert pool1.pool_size == (2, 2)
        
        # Check Dense layers
        dense1 = model.layers[3]  # After Flatten
        assert dense1.units == 128
        
        dense2 = model.layers[4]  # Output layer
        assert dense2.units == 10
    
    def test_model_default_num_classes(self):
        """Test default num_classes behavior"""
        # For MNIST/EMNIST_DIGITS: should default to 10 if None
        model_mnist = ModelFactory.create_model(DatasetType.MNIST, num_classes=None, seed=42)
        assert model_mnist.output_shape == (None, 10)
        
        # For EMNIST_CHAR: should default to 62 if None  
        model_emnist = ModelFactory.create_model(DatasetType.EMNIST_CHAR, num_classes=None, seed=42)
        assert model_emnist.output_shape == (None, 62)
        
        # For CIFAR10: should default to 10 if None
        model_cifar = ModelFactory.create_model(DatasetType.CIFAR10, num_classes=None, seed=42)
        assert model_cifar.output_shape == (None, 10)


class TestAggregationStrategy:
    """Test abstract AggregationStrategy class"""
    
    def test_is_abstract(self):
        """Test that AggregationStrategy cannot be instantiated"""
        with pytest.raises(TypeError):
            AggregationStrategy()


class TestFedAvgAggregation:
    """Test FedAvgAggregation class"""
    
    def test_aggregate_empty_weights(self):
        aggregator = FedAvgAggregation()
        
        result = aggregator.aggregate([], [])
        assert result is None
    
    def test_aggregate_single_client(self):
        aggregator = FedAvgAggregation()
        
        # Create mock weights for single client
        client_weights = [
            [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])]
        ]
        client_samples = [10]
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Should return the same weights
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], client_weights[0][0])
        np.testing.assert_array_equal(result[1], client_weights[0][1])
    
    def test_aggregate_multiple_clients_equal_samples(self):
        aggregator = FedAvgAggregation()
        
        # Create weights for two clients with equal samples
        client_weights = [
            [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])],
            [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.0, 2.0])]
        ]
        client_samples = [10, 10]  # Equal samples
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Should be simple average since equal samples
        expected_layer1 = (client_weights[0][0] + client_weights[1][0]) / 2
        expected_layer2 = (client_weights[0][1] + client_weights[1][1]) / 2
        
        np.testing.assert_array_almost_equal(result[0], expected_layer1)
        np.testing.assert_array_almost_equal(result[1], expected_layer2)
    
    def test_aggregate_multiple_clients_different_samples(self):
        aggregator = FedAvgAggregation()
        
        # Create weights for two clients with different samples
        client_weights = [
            [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])],
            [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.0, 2.0])]
        ]
        client_samples = [30, 10]  # 3:1 ratio
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Should be weighted average: (30*w1 + 10*w2) / 40 = 0.75*w1 + 0.25*w2
        expected_layer1 = 0.75 * client_weights[0][0] + 0.25 * client_weights[1][0]
        expected_layer2 = 0.75 * client_weights[0][1] + 0.25 * client_weights[1][1]
        
        np.testing.assert_array_almost_equal(result[0], expected_layer1)
        np.testing.assert_array_almost_equal(result[1], expected_layer2)
    
    def test_aggregate_with_zero_samples(self):
        aggregator = FedAvgAggregation()
        
        # One client has zero samples
        client_weights = [
            [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])],
            [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.0, 2.0])]
        ]
        client_samples = [10, 0]  # Second client has no samples
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Should only use first client's weights
        np.testing.assert_array_equal(result[0], client_weights[0][0])
        np.testing.assert_array_equal(result[1], client_weights[0][1])
    
    def test_aggregate_all_zero_samples(self):
        aggregator = FedAvgAggregation()
        
        client_weights = [
            [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])],
            [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([1.0, 2.0])]
        ]
        client_samples = [0, 0]  # All clients have no samples
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Should return first client's weights as fallback
        np.testing.assert_array_equal(result[0], client_weights[0][0])
        np.testing.assert_array_equal(result[1], client_weights[0][1])
    
    def test_aggregate_many_clients(self):
        aggregator = FedAvgAggregation()
        
        # Test with many clients
        num_clients = 5
        client_weights = []
        client_samples = []
        
        for i in range(num_clients):
            weights = [
                np.array([[i, i+1], [i+2, i+3]], dtype=float),
                np.array([i*0.1, (i+1)*0.1], dtype=float)
            ]
            client_weights.append(weights)
            client_samples.append((i + 1) * 10)  # 10, 20, 30, 40, 50 samples
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Verify result has correct structure
        assert len(result) == 2
        assert result[0].shape == (2, 2)
        assert result[1].shape == (2,)
        
        # Verify weighted average calculation manually for first element
        total_samples = sum(client_samples)
        expected_00 = sum(
            client_weights[i][0][0, 0] * client_samples[i] 
            for i in range(num_clients)
        ) / total_samples
        
        assert abs(result[0][0, 0] - expected_00) < 1e-10
    
    def test_aggregate_different_dtypes(self):
        aggregator = FedAvgAggregation()
        
        # Test with different numpy dtypes
        client_weights = [
            [np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([0.5, 1.5], dtype=np.float32)],
            [np.array([[2, 3], [4, 5]], dtype=np.float32), np.array([1.0, 2.0], dtype=np.float32)]
        ]
        client_samples = [10, 10]
        
        result = aggregator.aggregate(client_weights, client_samples)
        
        # Should maintain dtype
        assert result[0].dtype == np.float32
        assert result[1].dtype == np.float32


class TestAggregationFactory:
    """Test AggregationFactory class"""
    
    def test_create_aggregator_fedavg(self):
        aggregator = AggregationFactory.create_aggregator(AggregationMethod.FEDAVG)
        
        assert isinstance(aggregator, FedAvgAggregation)
        assert isinstance(aggregator, AggregationStrategy)
    
    def test_create_aggregator_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            # Test with unsupported method
            AggregationFactory.create_aggregator(AggregationMethod.FEDPROX)
    
    def test_create_aggregator_invalid_type(self):
        with pytest.raises(ValueError):
            # Test with completely invalid type
            AggregationFactory.create_aggregator("invalid_method")


class TestModelsIntegration:
    """Integration tests for models module"""
    
    def test_model_with_aggregation_workflow(self):
        """Test a complete workflow of model creation and aggregation"""
        
        # Create two identical models
        model1 = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        model2 = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        
        # Modify weights slightly to simulate training
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        
        # Add small random changes to simulate training updates
        np.random.seed(42)
        for i in range(len(weights1)):
            weights1[i] = weights1[i] + np.random.normal(0, 0.01, weights1[i].shape)
            weights2[i] = weights2[i] + np.random.normal(0, 0.01, weights2[i].shape)
        
        # Aggregate the weights
        aggregator = AggregationFactory.create_aggregator(AggregationMethod.FEDAVG)
        client_weights = [weights1, weights2]
        client_samples = [100, 200]  # Different sample counts
        
        aggregated_weights = aggregator.aggregate(client_weights, client_samples)
        
        # Verify aggregated weights are different from original
        original_weights = model1.get_weights()  # Get fresh weights
        
        weights_changed = False
        for orig, agg in zip(original_weights, aggregated_weights):
            if not np.array_equal(orig, agg):
                weights_changed = True
                break
        
        assert weights_changed, "Aggregated weights should be different from original"
        
        # Verify aggregated weights can be set back to model
        model1.set_weights(aggregated_weights)
        
        # Model should work with aggregated weights
        test_input = np.random.random((1, 28, 28, 1))
        output = model1.predict(test_input, verbose=0)
        
        assert output.shape == (1, 10)
        assert np.isclose(np.sum(output), 1.0, atol=1e-5)  # Softmax output should sum to 1
    
    def test_model_compilation_and_training(self):
        """Test that created models can be compiled and trained"""
        
        model = ModelFactory.create_model(DatasetType.MNIST, num_classes=10, seed=42)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        x_train = np.random.random((32, 28, 28, 1))
        y_train = np.random.randint(0, 10, 32)
        
        # Should be able to train for one step
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        
        # Should have loss and accuracy in history
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 1
        assert len(history.history['accuracy']) == 1
    
    def test_model_serialization_compatibility(self):
        """Test that models can be serialized/deserialized for FL"""
        
        model = ModelFactory.create_model(DatasetType.CIFAR10, num_classes=10, seed=42)
        
        # Get weights
        original_weights = model.get_weights()
        
        # Simulate serialization by converting to list and back
        weights_as_lists = [w.tolist() for w in original_weights]
        restored_weights = [np.array(w) for w in weights_as_lists]
        
        # Should be able to restore weights
        model.set_weights(restored_weights)
        final_weights = model.get_weights()
        
        # Weights should be identical
        for orig, final in zip(original_weights, final_weights):
            np.testing.assert_array_equal(orig, final)