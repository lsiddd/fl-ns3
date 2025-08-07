"""
Data management classes for Federated Learning simulation.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
from typing import List, Tuple

from config import FLConfig, DatasetType, NonIIDType, QuantitySkewType, FeatureSkewType


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