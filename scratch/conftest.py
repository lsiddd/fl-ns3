"""
Pytest configuration and shared fixtures for FL API tests.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
import tensorflow as tf


@pytest.fixture(scope="session", autouse=True)
def setup_tensorflow():
    """Setup TensorFlow for testing"""
    # Disable GPU to avoid resource conflicts in tests
    tf.config.set_visible_devices([], 'GPU')
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # Set deterministic operations
    tf.config.experimental.enable_op_determinism()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility"""
    np.random.seed(42)
    tf.random.set_seed(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_dataset_samples():
    """Create mock dataset samples for testing"""
    train_samples = []
    test_samples = []
    
    # Create 100 training samples
    for i in range(100):
        train_samples.append({
            'image': np.random.randint(0, 255, (28, 28), dtype=np.uint8),
            'label': i % 10
        })
    
    # Create 20 test samples
    for i in range(20):
        test_samples.append({
            'image': np.random.randint(0, 255, (28, 28), dtype=np.uint8),
            'label': i % 10
        })
    
    return train_samples, test_samples


@pytest.fixture
def sample_model_weights():
    """Create sample model weights for testing"""
    return [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([0.5, 1.5], dtype=np.float32),
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        np.array([0.9], dtype=np.float32)
    ]


@pytest.fixture
def clear_tf_session():
    """Clear TensorFlow session after each test"""
    yield
    tf.keras.backend.clear_session()


# Skip tests that require actual dataset downloads if not available
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may download datasets)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_api" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (those that might download datasets)
        if any(keyword in item.nodeid.lower() for keyword in ["load_dataset", "mnist", "cifar"]):
            item.add_marker(pytest.mark.slow)