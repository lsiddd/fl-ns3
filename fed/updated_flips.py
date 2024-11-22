import gc
import matplotlib.pyplot as plt
import traceback
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def split_data(train_images: np.ndarray, train_labels: np.ndarray, n_clients: int, client_id: int, alpha: float = 2) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(client_id)
    unique_classes: np.ndarray = np.unique(train_labels)
    n_classes: int = len(unique_classes)

    # Get the indices of each class
    class_indices: Dict[Any, np.ndarray] = {cls: np.where(train_labels == cls)[0] for cls in unique_classes}

    # Use Dirichlet distribution to generate proportions for this client
    class_proportions: np.ndarray = np.random.dirichlet(alpha * np.ones(n_clients), n_classes)

    # Select data for the specific client_id
    client_indices: List[int] = []
    for cls in unique_classes:
        cls_indices: np.ndarray = class_indices[cls]
        np.random.shuffle(cls_indices)

        # Allocate data to the client based on the proportion
        n_samples_for_client: int = int(len(cls_indices) * class_proportions[cls, client_id])
        client_indices.extend(cls_indices[:n_samples_for_client])

    np.random.shuffle(client_indices)

    # Return the data for this client
    return train_images[client_indices], train_labels[client_indices]


def create_model() -> tf.keras.Model:
    model: tf.keras.Model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Dropout(0.25), tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Dropout(0.25), tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.25), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.25), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(10, activation="softmax")])
    return model


def get_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int = 64) -> tf.data.Dataset:
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model(model: tf.keras.Model, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, epochs: int = 1) -> tf.keras.callbacks.History:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2)


def quantize_weights(weights: List[np.ndarray], quantization_levels: int = 256) -> List[np.ndarray]:
    quantized_weights: List[np.ndarray] = []
    for layer_weights in weights:
        min_val: float = np.min(layer_weights)
        max_val: float = np.max(layer_weights)
        if max_val == min_val:
            quantized_layer: np.ndarray = layer_weights
        else:
            step_size: float = (max_val - min_val) / (quantization_levels - 1)
            quantized_layer = np.round((layer_weights - min_val) / step_size) * step_size + min_val
        quantized_weights.append(quantized_layer.astype(np.float32))
    return quantized_weights


def prune_weights(weights: List[np.ndarray], pruning_percentage: float = 0.2) -> List[np.ndarray]:
    pruned_weights: List[np.ndarray] = []
    for layer_weights in weights:
        if layer_weights.ndim > 0:
            threshold: float = np.percentile(np.abs(layer_weights), pruning_percentage * 100)
            pruned_layer_weights: np.ndarray = np.where(np.abs(layer_weights) < threshold, 0, layer_weights)
            pruned_weights.append(pruned_layer_weights)
        else:
            pruned_weights.append(layer_weights)
    return pruned_weights


def compress_weights(weights: List[np.ndarray]) -> List[bytes]:
    compressed_weights: List[bytes] = []
    for layer_weights in weights:
        weight_bytes: bytes = layer_weights.tobytes()
        compressed_layer_weights: bytes = zlib.compress(weight_bytes)
        compressed_weights.append(compressed_layer_weights)
    return compressed_weights


def get_weights(model: tf.keras.Model) -> List[np.ndarray]:
    return model.get_weights()


def set_weights(model: tf.keras.Model, weights: List[np.ndarray]) -> None:
    model.set_weights(weights)


def average_weights(weights_list: List[List[np.ndarray]], weights_scaling_factors: Optional[List[float]] = None) -> List[np.ndarray]:
    if weights_scaling_factors is None:
        weights_scaling_factors: List[float] = [1.0 / len(weights_list)] * len(weights_list)
    avg_weights: List[np.ndarray] = []
    for weights in zip(*weights_list):
        weighted_sum: np.ndarray = sum(w * s for w, s in zip(weights, weights_scaling_factors))
        avg_weights.append(weighted_sum)
    return avg_weights


def fedprox_update(global_weights: List[np.ndarray], local_weights: List[np.ndarray], mu: float) -> List[np.ndarray]:
    updated_weights: List[np.ndarray] = []
    for gw, lw in zip(global_weights, local_weights):
        updated_weight: np.ndarray = lw - mu * (lw - gw)
        updated_weights.append(updated_weight)
    return updated_weights


def save_weights_to_file(filename: str, compressed_weights_list: List[bytes]) -> None:
    with open(filename, "wb") as f:
        for compressed_weights in compressed_weights_list:
            f.write(compressed_weights)


def save_compressed_top_n_layers(model: tf.keras.Model, top_n_layers: List[tf.keras.layers.Layer], output_filename: str) -> int:
    try:
        new_model: tf.keras.Model = tf.keras.models.Sequential(top_n_layers)
    except Exception:
        new_model = model
    new_model_weights: List[np.ndarray] = get_weights(new_model)
    new_quantized_weights: List[np.ndarray] = quantize_weights(new_model_weights)
    new_compressed_weights: List[bytes] = compress_weights(new_quantized_weights)
    compressed_size: int = sum(len(cw) for cw in new_compressed_weights)
    save_weights_to_file(output_filename, new_compressed_weights)
    return compressed_size


def rank_model_layers(model: tf.keras.Model, test_dataset: tf.data.Dataset, subset_size: int = 1000, batch_size: int = 64) -> List[Tuple[int, str, float]]:
    # Use a subset of the test dataset
    test_dataset_subset: tf.data.Dataset = test_dataset.take(subset_size // batch_size)
    base_accuracy: float = model.evaluate(test_dataset_subset, verbose=0)[1]

    def evaluate_layer_impact(i: int, layer: tf.keras.layers.Layer) -> Tuple[int, str, float]:
        original_weights: List[np.ndarray] = layer.get_weights()
        if not original_weights:
            return i, layer.name, 0.0

        zeroed_weights: List[np.ndarray] = [np.zeros_like(w) for w in original_weights]
        layer.set_weights(zeroed_weights)
        perturbed_accuracy: float = model.evaluate(test_dataset_subset, verbose=0)[1]
        accuracy_drop: float = base_accuracy - perturbed_accuracy
        layer.set_weights(original_weights)

        del original_weights
        del zeroed_weights

        return i, layer.name, accuracy_drop

    layer_impact: List[Tuple[int, str, float]] = []
    for i, layer in enumerate(model.layers):
        result: Tuple[int, str, float] = evaluate_layer_impact(i, layer)
        layer_impact.append(result)

    layer_impact.sort(key=lambda x: x[2], reverse=True)
    return layer_impact


def client_update(client_id: int, global_weights: List[np.ndarray], train_images: np.ndarray, train_labels: np.ndarray, test_dataset: tf.data.Dataset, aggregation_method: str, **kwargs) -> Tuple[List[np.ndarray], int, Optional[List[np.ndarray]]]:
    # Each client starts with the global model
    client_model: tf.keras.Model = create_model()
    set_weights(client_model, global_weights)

    # Split data for client
    train_images_chunk: np.ndarray
    train_labels_chunk: np.ndarray
    train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, kwargs["n_clients"], client_id)
    client_samples: int = len(train_images_chunk)

    # Prepare train_dataset
    train_dataset: tf.data.Dataset = get_dataset(train_images_chunk, train_labels_chunk)

    if aggregation_method == "SCAFFOLD":
        # Need client_control_variate and server_control_variate
        client_cv: List[np.ndarray] = kwargs["client_control_variates"][client_id]
        server_cv: List[np.ndarray] = kwargs["server_control_variate"]

        # Custom optimizer for SCAFFOLD
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        @tf.function
        def train_step(images: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as tape:
                predictions = client_model(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, client_model.trainable_variables)

            # Adjust gradients with control variates
            adjusted_grads = [g - c + s for g, c, s in zip(grads, client_cv, server_cv)]

            optimizer.apply_gradients(zip(adjusted_grads, client_model.trainable_variables))
            return loss

        # Custom training loop
        epochs: int = kwargs.get("epochs", 1)
        for epoch in range(epochs):
            for images, labels in train_dataset:
                loss = train_step(images, labels)

        # Extract trainable variables
        trainable_global_weights: List[np.ndarray] = [v.numpy() for v in client_model.trainable_variables]

        # After training
        local_weights: List[np.ndarray] = client_model.get_weights()
        trainable_local_weights: List[np.ndarray] = [v.numpy() for v in client_model.trainable_variables]

        # Update client control variate
        new_client_cv: List[np.ndarray] = []
        for lw, gw, cc, sc in zip(trainable_local_weights, trainable_global_weights, client_cv, server_cv):
            delta_w: np.ndarray = lw - gw
            new_cc: np.ndarray = cc - sc + delta_w / (epochs * optimizer.learning_rate)
            new_client_cv.append(new_cc)
    else:
        # For 'FedAvg' and 'FedProx', train normally
        epochs: int = kwargs.get("epochs", 1)
        history: tf.keras.callbacks.History = train_model(client_model, train_dataset, test_dataset, epochs=epochs)
        local_weights: List[np.ndarray] = client_model.get_weights()
        new_client_cv = None  # Not applicable

    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()

    return local_weights, client_samples, new_client_cv


def aggregate_models(client_weights: List[List[np.ndarray]], client_samples: List[int], global_weights: List[np.ndarray], aggregation_method: str, global_model: Optional[tf.keras.Model] = None, **kwargs) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    if aggregation_method == "FedAvg":
        # Weight by number of samples
        total_samples: int = sum(client_samples)
        scaling_factors: List[float] = [num / total_samples for num in client_samples]
        new_global_weights: List[np.ndarray] = average_weights(client_weights, scaling_factors)
        return new_global_weights
    elif aggregation_method == "FedProx":
        total_samples: int = sum(client_samples)
        scaling_factors: List[float] = [num / total_samples for num in client_samples]
        avg_weights: List[np.ndarray] = average_weights(client_weights, scaling_factors)
        mu: float = kwargs.get("mu", 0.01)
        new_global_weights: List[np.ndarray] = fedprox_update(global_weights, avg_weights, mu)
        return new_global_weights
    elif aggregation_method == "SCAFFOLD":
        # Extract trainable weights from client weights
        client_trainable_weights: List[List[np.ndarray]] = [w[: len(global_model.trainable_variables)] for w in client_weights]

        # Aggregate trainable weights
        total_samples: int = sum(client_samples)
        scaling_factors: List[float] = [num / total_samples for num in client_samples]
        new_global_trainable_weights: List[np.ndarray] = average_weights(client_trainable_weights, scaling_factors)

        # Update server control variate
        delta_c: List[np.ndarray] = []
        client_control_variates: List[List[np.ndarray]] = kwargs["client_control_variates"]
        server_control_variate: List[np.ndarray] = kwargs["server_control_variate"]
        n_clients: int = len(client_weights)
        for idx in range(len(server_control_variate)):
            delta: np.ndarray = sum(scaling_factors[i] * (client_control_variates[i][idx] - server_control_variate[idx]) for i in range(n_clients))
            delta_c.append(delta)
        updated_server_control_variate: List[np.ndarray] = [sc + dc for sc, dc in zip(server_control_variate, delta_c)]

        # Set new global weights
        new_global_weights: List[np.ndarray] = global_weights.copy()
        new_global_weights[: len(new_global_trainable_weights)] = new_global_trainable_weights
        return new_global_weights, updated_server_control_variate
    else:
        raise ValueError("Invalid aggregation method")


def flatten(xss: List[List[Any]]) -> List[Any]:
    return [x for xs in xss for x in xs]


def flips_algorithm(train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray, n_clients: int, n_rounds: int, epochs: int, pruning_ratio: float = 0.2) -> tf.keras.Model:

    # Initialize global model
    global_model: tf.keras.Model = create_model()
    global_weights: List[np.ndarray] = global_model.get_weights()

    # Prepare test dataset
    test_dataset: tf.data.Dataset = get_dataset(test_images, test_labels)

    def evaluate_layer_importance(client_model: tf.keras.Model, validation_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluates layer importance using SHAP-like methodology and maps scores to layer names.
        """
        base_accuracy: float = client_model.evaluate(validation_data, verbose=0)[1]
        importance_scores: Dict[str, float] = {}

        for layer in client_model.layers:
            original_weights: List[np.ndarray] = layer.get_weights()
            if not original_weights:  # Skip layers without weights
                importance_scores[layer.name] = 0.0
                continue

            # Zero out layer weights
            zeroed_weights: List[np.ndarray] = [np.zeros_like(w) for w in original_weights]
            layer.set_weights(zeroed_weights)

            # Evaluate model performance with zeroed weights
            perturbed_accuracy: float = client_model.evaluate(validation_data, verbose=0)[1]
            importance_score: float = base_accuracy - perturbed_accuracy
            importance_scores[layer.name] = importance_score

            # Restore original weights
            layer.set_weights(original_weights)

        # Normalize importance scores
        total_importance: float = sum(importance_scores.values())
        for layer_name in importance_scores:
            if total_importance > 0:
                importance_scores[layer_name] /= total_importance
            else:
                importance_scores[layer_name] = 0.0

        return importance_scores

    def prune_model(model: tf.keras.Model, importance_scores: Dict[str, float], pruning_ratio: float) -> None:
        return

        weights: List[np.ndarray] = model.get_weights()
        pruned_weights: List[np.ndarray] = []

        for i, layer_weights in enumerate(weights):
            if len(layer_weights.shape) == 0:  # Skip layers without weights
                pruned_weights.append(layer_weights)
                continue

            # Get importance score for this layer
            importance_score: float = importance_scores[i] if i < len(importance_scores) else 0.1

            # Calculate per-layer pruning ratio
            per_layer_pruning_ratio: float = pruning_ratio * (1 - importance_score)

            # Prune weights below the threshold
            threshold: float = np.percentile(np.abs(layer_weights), per_layer_pruning_ratio * 100)
            layer_weights[np.abs(layer_weights) < threshold] = 0

            pruned_weights.append(layer_weights)

        model.set_weights(pruned_weights)

    def aggregate_models(client_weights: List[List[np.ndarray]], client_samples: List[int], global_weights: List[np.ndarray], layer_importance: Dict[str, float], global_model: tf.keras.Model, learning_rate: float = 0.1) -> List[np.ndarray]:
        # Get layer names from the global model
        layer_names: List[str] = [layer.name for layer in global_model.layers]
        total_samples: int = sum(client_samples)
        new_weights: List[np.ndarray] = []

        for layer_index, (layer_name, layer_weights) in enumerate(zip(layer_names, zip(*client_weights))):
            # Get the importance score for this layer
            # Default importance if not found
            importance: float = layer_importance.get(layer_name, 0.1)

            # Weighted average of the layer's weights from clients
            weighted_average: np.ndarray = sum(client_layer * (samples / total_samples) * importance for client_layer, samples in zip(layer_weights, client_samples))

            # Blend current global weights with the weighted average
            updated_layer: np.ndarray = (1 - learning_rate) * global_weights[layer_index] + learning_rate * weighted_average
            new_weights.append(updated_layer)

        return new_weights

    # Run communication rounds
    for round_num in range(n_rounds):
        print(f"--- Round {round_num + 1} ---")
        client_weights: List[List[np.ndarray]] = []
        client_samples: List[int] = []
        layer_importance_scores: List[Dict[str, float]] = []

        # Simulate clients
        for client_id in range(n_clients):
            print(f"Client {client_id + 1}/{n_clients}")
            # Split data for client
            train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, n_clients, client_id)

            # Split client's data into training and validation sets
            (train_images_train, train_images_val, train_labels_train, train_labels_val) = train_test_split(train_images_chunk, train_labels_chunk, test_size=0.2, random_state=client_id)

            optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

            train_dataset: tf.data.Dataset = get_dataset(train_images_train, train_labels_train)
            val_dataset: tf.data.Dataset = get_dataset(train_images_val, train_labels_val)

            # Initialize and set client model
            client_model: tf.keras.Model = create_model()
            client_model.set_weights(global_weights)

            client_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # Train client model locally
            client_model.fit(train_dataset, epochs=epochs, verbose=0)

            # Evaluate layer importance using client's validation data
            importance_scores: Dict[str, float] = evaluate_layer_importance(client_model, val_dataset)
            layer_importance_scores.append(importance_scores)

            # Prune model based on importance
            prune_model(client_model, importance_scores, pruning_ratio)

            # Collect client weights and sample count
            client_weights.append(client_model.get_weights())
            client_samples.append(len(train_labels_chunk))

        # Average layer importance across clients
        avg_layer_importance: Dict[str, float] = {}
        total_clients: int = len(layer_importance_scores)
        for importance_scores in layer_importance_scores:
            for layer_name, importance in importance_scores.items():
                avg_layer_importance[layer_name] = avg_layer_importance.get(layer_name, 0.0) + importance / total_clients

        # Aggregate models
        global_weights = aggregate_models(client_weights, client_samples, global_weights, avg_layer_importance, global_model)
        global_model.set_weights(global_weights)

        # Evaluate global model
        global_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Global Model Accuracy: {accuracy:.4f}")

    return global_model


def main() -> None:
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images: np.ndarray = train_images[..., np.newaxis] / 255.0
    test_images: np.ndarray = test_images[..., np.newaxis] / 255.0

    # Create test_dataset
    test_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64).prefetch(tf.data.AUTOTUNE)

    n_clients: int = 4
    n_rounds: int = 50
    epochs: int = 1  # Increased local epochs per round
    # Choose from 'FedAvg', 'FedProx', 'SCAFFOLD', 'flips'
    aggregation_method: str = "flips"
    mu: float = 0.01  # Proximal term coefficient for FedProx
    pruning_ratio: float = 0  # Set an appropriate pruning ratio

    global_model: tf.keras.Model = create_model()
    global_weights: List[np.ndarray] = global_model.get_weights()
    trainable_variables: List[tf.Variable] = global_model.trainable_variables

    if aggregation_method == "SCAFFOLD":
        server_control_variate: List[tf.Variable] = [tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_variables]
        client_control_variates: List[List[tf.Variable]] = [[tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_variables] for _ in range(n_clients)]
    elif aggregation_method == "flips":
        # Run FLIPS algorithm with improved strategies
        final_model: tf.keras.Model = flips_algorithm(train_images, train_labels, test_images, test_labels, n_clients, n_rounds, epochs, pruning_ratio)
        final_model.save("flips_global_model.h5")
        print("\nTraining complete. Global model saved as 'flips_global_model.h5'")
        return

    else:
        server_control_variate = None
        client_control_variates = None

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num+1} ---")
        client_weights: List[List[np.ndarray]] = []
        client_samples: List[int] = []
        for client_id in range(n_clients):
            print(f"\nClient {client_id}")
            try:
                local_weights: List[np.ndarray]
                samples: int
                new_client_cv: Optional[List[np.ndarray]]
                local_weights, samples, new_client_cv = client_update(client_id, global_weights, train_images, train_labels, test_dataset, aggregation_method, n_clients=n_clients, client_control_variates=client_control_variates, server_control_variate=server_control_variate, epochs=epochs, mu=mu)
                client_weights.append(local_weights)
                client_samples.append(samples)
                if aggregation_method == "SCAFFOLD" and new_client_cv is not None:
                    client_control_variates[client_id] = new_client_cv
            except Exception as e:
                print(f"Error processing client {client_id}: {e}")
                traceback.print_exc()

        # Aggregate client models
        print("\nAggregating client models")
        if aggregation_method == "SCAFFOLD":
            global_weights, server_control_variate = aggregate_models(client_weights, client_samples, global_weights, aggregation_method, global_model=global_model, client_control_variates=client_control_variates, server_control_variate=server_control_variate)
        else:
            global_weights = aggregate_models(client_weights, client_samples, global_weights, aggregation_method, mu=mu)

        # Update global model
        global_model.set_weights(global_weights)

        # Evaluate global model
        global_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loss: float
        accuracy: float
        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Round {round_num+1} Global model accuracy: {accuracy:.4f}")

    # Save final global model
    global_model.save("global_model.h5")
    print("\nTraining complete. Global model saved as 'global_model.h5'")


if __name__ == "__main__":
    main()
