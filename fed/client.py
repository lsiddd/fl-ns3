import gc
import json
import os
import time
import traceback
import zlib

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Set environment variable to use CPU only
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def split_data(train_images, train_labels, n_clients, client_id):
    np.random.seed(client_id)
    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)
    data_size = len(train_images)

    # Total number of samples per client
    samples_per_client = data_size // n_clients

    # Define mean and standard deviation for Gaussian distribution over classes
    # Spread mu from 0 to n_classes - 1 across clients
    mu = client_id * (n_classes - 1) / (n_clients - 1)
    sigma = n_classes / 2  # Adjust sigma to control the spread

    # Compute class probabilities using Gaussian distribution
    class_labels = np.arange(n_classes)
    class_probabilities = np.exp(-((class_labels - mu) ** 2) / (2 * sigma**2))
    class_probabilities += 1e-6  # Avoid zeros
    class_probabilities /= np.sum(class_probabilities)  # Normalize to sum to 1

    # Compute number of samples per class for this client
    num_samples_per_class = (class_probabilities * samples_per_client).astype(int)

    # Ensure at least one sample per class
    num_samples_per_class[num_samples_per_class == 0] = 1

    # Adjust total number of samples to match samples_per_client
    total_samples = np.sum(num_samples_per_class)
    while total_samples > samples_per_client:
        # Reduce samples from the class with the most samples
        max_class = np.argmax(num_samples_per_class)
        num_samples_per_class[max_class] -= 1
        total_samples -= 1
    while total_samples < samples_per_client:
        # Add samples to the class with the most samples
        max_class = np.argmax(num_samples_per_class)
        num_samples_per_class[max_class] += 1
        total_samples += 1

    # Collect indices for each class
    client_indices = []
    for cls, num_samples in enumerate(num_samples_per_class):
        class_indices = np.where(train_labels == cls)[0]
        np.random.shuffle(class_indices)
        # Ensure we don't sample more than available
        num_samples = min(num_samples, len(class_indices))
        selected_indices = class_indices[:num_samples]
        client_indices.extend(selected_indices)

    np.random.shuffle(client_indices)

    # Return the data for this client
    return train_images[client_indices], train_labels[client_indices]


def create_model():
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Dropout(0.25), tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"), tf.keras.layers.BatchNormalization(), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Dropout(0.25), tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.25), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dropout(0.25), tf.keras.layers.BatchNormalization(), tf.keras.layers.Dense(10, activation="softmax")])
    return model


def get_dataset(images, labels, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_model(model, train_dataset, test_dataset, epochs=1):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2)


def quantize_weights(weights, quantization_levels=256):
    quantized_weights = []
    for layer_weights in weights:
        min_val = np.min(layer_weights)
        max_val = np.max(layer_weights)
        if max_val == min_val:
            quantized_layer = layer_weights
        else:
            step_size = (max_val - min_val) / (quantization_levels - 1)
            quantized_layer = np.round((layer_weights - min_val) / step_size) * step_size + min_val
        quantized_weights.append(quantized_layer.astype(np.float32))
    return quantized_weights


def prune_weights(weights, pruning_percentage=0.2):
    pruned_weights = []
    for layer_weights in weights:
        if layer_weights.ndim > 0:
            threshold = np.percentile(np.abs(layer_weights), pruning_percentage * 100)
            pruned_layer_weights = np.where(np.abs(layer_weights) < threshold, 0, layer_weights)
            pruned_weights.append(pruned_layer_weights)
        else:
            pruned_weights.append(layer_weights)
    return pruned_weights


def compress_weights(weights):
    compressed_weights = []
    for layer_weights in weights:
        weight_bytes = layer_weights.tobytes()
        compressed_layer_weights = zlib.compress(weight_bytes)
        compressed_weights.append(compressed_layer_weights)
    return compressed_weights


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def average_weights(weights_list, weights_scaling_factors=None):
    if weights_scaling_factors is None:
        weights_scaling_factors = [1.0 / len(weights_list)] * len(weights_list)
    avg_weights = []
    for weights in zip(*weights_list):
        weighted_sum = sum(w * s for w, s in zip(weights, weights_scaling_factors))
        avg_weights.append(weighted_sum)
    return avg_weights


def fedprox_update(global_weights, local_weights, mu):
    updated_weights = []
    for gw, lw in zip(global_weights, local_weights):
        updated_weights.append(lw - mu * (lw - gw))
    return updated_weights


def save_weights_to_file(filename, compressed_weights_list):
    with open(filename, "wb") as f:
        for compressed_weights in compressed_weights_list:
            f.write(compressed_weights)


def save_compressed_top_n_layers(model, top_n_layers, output_filename):
    try:
        new_model = tf.keras.models.Sequential(top_n_layers)
    except:
        new_model = model
    new_model_weights = get_weights(new_model)
    new_quantized_weights = quantize_weights(new_model_weights)
    new_compressed_weights = compress_weights(new_quantized_weights)
    compressed_size = sum(len(cw) for cw in new_compressed_weights)
    save_weights_to_file(output_filename, new_compressed_weights)
    return compressed_size


def rank_model_layers(model, test_dataset, subset_size=1000, batch_size=64):
    # Use a subset of the test dataset
    test_dataset_subset = test_dataset.take(subset_size // batch_size)
    base_accuracy = model.evaluate(test_dataset_subset, verbose=0)[1]

    def evaluate_layer_impact(i, layer):
        original_weights = layer.get_weights()
        if not original_weights:
            return i, layer.name, 0

        zeroed_weights = [np.zeros_like(w) for w in original_weights]
        layer.set_weights(zeroed_weights)
        perturbed_accuracy = model.evaluate(test_dataset_subset, verbose=0)[1]
        accuracy_drop = base_accuracy - perturbed_accuracy
        layer.set_weights(original_weights)

        del original_weights
        del zeroed_weights

        return i, layer.name, accuracy_drop

    layer_impact = []
    for i, layer in enumerate(model.layers):
        result = evaluate_layer_impact(i, layer)
        layer_impact.append(result)

    layer_impact.sort(key=lambda x: x[2], reverse=True)
    return layer_impact


def client_update(client_id, global_weights, train_images, train_labels, test_dataset, aggregation_method, **kwargs):
    # Each client starts with the global model
    client_model = create_model()
    set_weights(client_model, global_weights)

    # Split data for client
    train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, kwargs["n_clients"], client_id)
    client_samples = len(train_images_chunk)

    # Prepare train_dataset
    train_dataset = get_dataset(train_images_chunk, train_labels_chunk)

    if aggregation_method == "SCAFFOLD":
        # Need client_control_variate and server_control_variate
        client_cv = kwargs["client_control_variates"][client_id]
        server_cv = kwargs["server_control_variate"]

        # Custom optimizer for SCAFFOLD
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

        @tf.function
        def train_step(images, labels):
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
        epochs = kwargs.get("epochs", 1)
        for epoch in range(epochs):
            for images, labels in train_dataset:
                loss = train_step(images, labels)

            # Extract trainable variables
        trainable_global_weights = [v.numpy() for v in client_model.trainable_variables]

        # After training
        local_weights = client_model.get_weights()
        trainable_local_weights = [v.numpy() for v in client_model.trainable_variables]

        # Update client control variate
        new_client_cv = []
        for lw, gw, cc, sc in zip(trainable_local_weights, trainable_global_weights, client_cv, server_cv):
            delta_w = lw - gw
            new_cc = cc - sc + delta_w / (epochs * optimizer.learning_rate)
            new_client_cv.append(new_cc)
    else:
        # For 'FedAvg' and 'FedProx', train normally
        epochs = kwargs.get("epochs", 1)
        history = train_model(client_model, train_dataset, test_dataset, epochs=epochs)
        local_weights = client_model.get_weights()
        new_client_cv = None  # Not applicable

    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()

    return local_weights, client_samples, new_client_cv


def aggregate_models(client_weights, client_samples, global_weights, aggregation_method, global_model=None, **kwargs):
    if aggregation_method == "FedAvg":
        # Weight by number of samples
        total_samples = sum(client_samples)
        scaling_factors = [num / total_samples for num in client_samples]
        new_global_weights = average_weights(client_weights, scaling_factors)
        return new_global_weights
    elif aggregation_method == "FedProx":
        total_samples = sum(client_samples)
        scaling_factors = [num / total_samples for num in client_samples]
        avg_weights = average_weights(client_weights, scaling_factors)
        mu = kwargs.get("mu", 0.01)
        new_global_weights = fedprox_update(global_weights, avg_weights, mu)
        return new_global_weights
    elif aggregation_method == "SCAFFOLD":
        # Extract trainable weights from client weights
        client_trainable_weights = [w[: len(global_model.trainable_variables)] for w in client_weights]

        # Aggregate trainable weights
        total_samples = sum(client_samples)
        scaling_factors = [num / total_samples for num in client_samples]
        new_global_trainable_weights = average_weights(client_trainable_weights, scaling_factors)

        # Update server control variate
        delta_c = []
        client_control_variates = kwargs["client_control_variates"]
        server_control_variate = kwargs["server_control_variate"]
        n_clients = len(client_weights)
        for idx in range(len(server_control_variate)):
            delta = sum(scaling_factors[i] * (client_control_variates[i][idx] - server_control_variate[idx]) for i in range(n_clients))
            delta_c.append(delta)
        updated_server_control_variate = [sc + dc for sc, dc in zip(server_control_variate, delta_c)]

        # Set new global weights
        new_global_weights = global_weights.copy()
        new_global_weights[: len(new_global_trainable_weights)] = new_global_trainable_weights
        return new_global_weights, updated_server_control_variate
    else:
        raise ValueError("Invalid aggregation method")


def flatten(xss):
    return [x for xs in xss for x in xs]


def flips_algorithm(train_images, train_labels, test_images, test_labels, n_clients, n_rounds, epochs, pruning_ratio=0.2):
    """
    Implements the FLIPS-inspired algorithm with selective layer pruning, SHAP-based importance,
    and importance-weighted aggregation.

    Args:
        train_images (ndarray): Training images.
        train_labels (ndarray): Training labels.
        test_images (ndarray): Test images.
        test_labels (ndarray): Test labels.
        n_clients (int): Number of clients.
        n_rounds (int): Number of communication rounds.
        epochs (int): Local training epochs per client.
        pruning_ratio (float): Ratio of weights to prune.
    Returns:
        tf.keras.Model: Trained global model.
    """
    # Initialize global model
    global_model = create_model()
    global_weights = global_model.get_weights()

    # Prepare test dataset
    test_dataset = get_dataset(test_images, test_labels)

    def evaluate_layer_importance(client_model, validation_data):
        """
        Evaluates layer importance using SHAP-like methodology for pruning.
        """
        base_accuracy = client_model.evaluate(validation_data, verbose=0)[1]

        importance_scores = []
        for layer_index, layer in enumerate(client_model.layers):
            original_weights = layer.get_weights()
            if not original_weights:  # Skip layers without weights
                importance_scores.append(0)
                continue

            # Zero out layer weights
            zeroed_weights = [np.zeros_like(w) for w in original_weights]
            layer.set_weights(zeroed_weights)

            # Evaluate model performance with zeroed weights
            perturbed_accuracy = client_model.evaluate(validation_data, verbose=0)[1]
            importance_score = base_accuracy - perturbed_accuracy
            importance_scores.append(importance_score)

            # Restore original weights
            layer.set_weights(original_weights)

        # Normalize importance scores
        total_importance = sum(importance_scores)
        normalized_scores = [score / total_importance if total_importance > 0 else 0 for score in importance_scores]

        return normalized_scores

    def prune_model(model, importance_scores, pruning_ratio):
        """
        Prunes model layers based on importance scores.
        Skips layers without weights.
        """
        weights = model.get_weights()
        pruned_weights = []

        for i, layer_weights in enumerate(weights):
            if len(layer_weights.shape) == 0:  # Skip layers without weights
                pruned_weights.append(layer_weights)
                continue

            # Check if the importance score exists for this layer
            importance_score = importance_scores[i] if i < len(importance_scores) else 0

            # Prune based on importance
            threshold = np.percentile(np.abs(layer_weights), pruning_ratio * 100)
            if importance_score < 0.5:  # Prune more aggressively for less important layers
                layer_weights[np.abs(layer_weights) < threshold] = 0
            pruned_weights.append(layer_weights)

        model.set_weights(pruned_weights)

    def aggregate_models(client_weights, client_samples, global_weights, layer_importance):
        """
        Aggregates client models using importance-weighted averaging.
        """
        total_samples = sum(client_samples)
        new_weights = []

        for layer_index, layer_weights in enumerate(zip(*client_weights)):
            # Check if the layer index exists in layer_importance
            importance = layer_importance[layer_index] if layer_index < len(layer_importance) else 0.1

            # Weighted sum of the layer's weights
            weighted_sum = sum(client_layer * (samples / total_samples) * importance for client_layer, samples in zip(layer_weights, client_samples))
            new_weights.append(weighted_sum)

        return new_weights

    # Run communication rounds
    for round_num in range(n_rounds):
        print(f"--- Round {round_num + 1} ---")
        client_weights = []
        client_samples = []
        layer_importance_scores = []

        # Simulate clients
        for client_id in range(n_clients):
            print(f"Client {client_id + 1}/{n_clients}")
            # Split data for client
            train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, n_clients, client_id)
            train_dataset = get_dataset(train_images_chunk, train_labels_chunk)

            # Initialize and set client model
            client_model = create_model()
            client_model.set_weights(global_weights)

            client_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            # Train client model locally
            client_model.fit(train_dataset, epochs=epochs, verbose=0)

            # Evaluate layer importance
            importance_scores = evaluate_layer_importance(client_model, test_dataset)
            layer_importance_scores.append(importance_scores)

            # Prune model based on importance
            prune_model(client_model, importance_scores, pruning_ratio)

            # Collect client weights and sample count
            client_weights.append(client_model.get_weights())
            client_samples.append(len(train_labels_chunk))

        # Average layer importance across clients
        avg_layer_importance = np.mean(flatten([list(i.values()) for i in layer_importance_scores]), axis=0)

        # Aggregate models
        global_weights = aggregate_models(client_weights, client_samples, global_weights, avg_layer_importance)
        global_model.set_weights(global_weights)

        # Evaluate global model
        global_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Global Model Accuracy: {accuracy:.4f}")

    return global_model


def main():
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images[..., np.newaxis] / 255.0
    test_images = test_images[..., np.newaxis] / 255.0

    # Create test_dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64).prefetch(tf.data.AUTOTUNE)

    n_clients = 4
    n_rounds = 5
    epochs = 1  # Local epochs per round
    aggregation_method = "flips"  # Choose from 'FedAvg', 'FedProx', 'SCAFFOLD'
    mu = 0.01  # Proximal term coefficient for FedProx
    pruning_ratio = 0

    global_model = create_model()
    global_weights = global_model.get_weights()
    trainable_variables = global_model.trainable_variables

    if aggregation_method == "SCAFFOLD":
        server_control_variate = [tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_variables]
        client_control_variates = [[tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_variables] for _ in range(n_clients)]
    elif aggregation_method == "flips":
        # Run FLIPS algorithm
        final_model = flips_algorithm(train_images, train_labels, test_images, test_labels, n_clients, n_rounds, epochs, pruning_ratio)
        final_model.save("flips_global_model.h5")
        return

    else:
        server_control_variate = None
        client_control_variates = None

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num+1} ---")
        client_weights = []
        client_samples = []
        for client_id in range(n_clients):
            print(f"\nClient {client_id}")
            try:
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
        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Round {round_num+1} Global model accuracy: {accuracy:.4f}")

    # Save final global model
    global_model.save("global_model.h5")
    print("\nTraining complete. Global model saved as 'global_model.h5'")


if __name__ == "__main__":
    main()
