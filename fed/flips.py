import gc
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Hyperparameters
n_clients: int = 5
n_rounds: int = 10
epochs: int = 1  # Increased local epochs per round
batch_size: int = 64
learning_rate: float = 0.001  # Adjusted learning rate
pruning_ratio: float = 0.2  # Adjusted pruning ratio

def split_data(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    n_clients: int,
    client_id: int,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    # Use Dirichlet distribution to create non-IID data splits
    np.random.seed(client_id)
    min_size = 0
    K = len(np.unique(train_labels))
    N = train_labels.shape[0]
    idx_batch = [[] for _ in range(n_clients)]
    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(K):
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

    client_indices = idx_batch[client_id]
    np.random.shuffle(client_indices)
    return train_images[client_indices], train_labels[client_indices]

def create_model() -> tf.keras.Model:
    model: tf.keras.Model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model

def get_dataset(
    images: np.ndarray, labels: np.ndarray, batch_size: int = 64
) -> tf.data.Dataset:
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = (
        dataset.shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset

def set_weights_by_layer_name(global_model: tf.keras.Model, aggregated_weights: Dict[str, List[np.ndarray]]) -> None:
    for layer in global_model.layers:
        if layer.name in aggregated_weights:
            layer_weights = aggregated_weights[layer.name]
            try:
                layer.set_weights(layer_weights)
            except ValueError as e:
                print(f"Failed to set weights for layer '{layer.name}': {e}")
        else:
            print(f"No aggregated weights found for layer '{layer.name}'.")

def aggregate_models(
    client_weights: List[Dict[str, List[np.ndarray]]],
    client_samples: List[int],
    global_model: tf.keras.Model,
) -> Dict[str, List[np.ndarray]]:
    aggregated_weights = {}

    for layer in global_model.layers:
        layer_name = layer.name

        # Collect weights and samples from clients that have this layer
        clients_with_layer = [
            (client[layer_name], samples)
            for client, samples in zip(client_weights, client_samples)
            if layer_name in client
        ]

        if not clients_with_layer:
            continue

        layer_weights_from_clients, samples_with_layer = zip(*clients_with_layer)
        num_weights = len(layer_weights_from_clients[0])
        aggregated_layer_weights = []

        for weight_index in range(num_weights):
            weights = np.array(
                [client_weights[weight_index] for client_weights in layer_weights_from_clients]
            )
            samples = np.array(samples_with_layer)

            # Ensure that samples array matches the weights array along axis=0
            aggregated_weight = np.average(weights, axis=0, weights=samples)
            aggregated_layer_weights.append(aggregated_weight)

        aggregated_weights[layer_name] = aggregated_layer_weights

    return aggregated_weights


def prune_model(model: tf.keras.Model, pruning_ratio: float) -> None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            original_weights = layer.get_weights()
            if not original_weights:
                continue
            pruned_weights = []
            for weight in original_weights:
                threshold = np.percentile(np.abs(weight), pruning_ratio * 100)
                weight[np.abs(weight) < threshold] = 0
                pruned_weights.append(weight)
            layer.set_weights(pruned_weights)

def flips_algorithm(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    n_clients: int,
    n_rounds: int,
    epochs: int,
    pruning_ratio: float,
) -> tf.keras.Model:
    # Initialize global model
    global_model: tf.keras.Model = create_model()
    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Prepare test dataset
    test_dataset: tf.data.Dataset = get_dataset(test_images, test_labels, batch_size)

    for round_num in range(n_rounds):
        print(f"--- Round {round_num + 1} ---")
        client_weights = []
        client_samples = []

        for client_id in range(n_clients):
            train_images_chunk, train_labels_chunk = split_data(
                train_images, train_labels, n_clients, client_id
            )
            train_dataset = get_dataset(train_images_chunk, train_labels_chunk, batch_size)

            client_model = create_model()
            client_model.set_weights(global_model.get_weights())

            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            client_model.fit(train_dataset, epochs=epochs, verbose=0)

            prune_model(client_model, pruning_ratio)

            # Collect weights as a dictionary mapping layer names to weights
            client_layer_weights = {}
            for layer in client_model.layers:
                client_layer_weights[layer.name] = layer.get_weights()
            client_weights.append(client_layer_weights)

            client_samples.append(len(train_labels_chunk))

            tf.keras.backend.clear_session()
            gc.collect()

        aggregated_weights = aggregate_models(
            client_weights, client_samples, global_model
        )

        set_weights_by_layer_name(global_model, aggregated_weights)

        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Global Model Accuracy: {accuracy:.4f}")

    return global_model

def main() -> None:
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images: np.ndarray = train_images[..., np.newaxis] / 255.0
    test_images: np.ndarray = test_images[..., np.newaxis] / 255.0

    # Run FLIPS algorithm
    final_model: tf.keras.Model = flips_algorithm(
        train_images,
        train_labels,
        test_images,
        test_labels,
        n_clients,
        n_rounds,
        epochs,
        pruning_ratio,
    )
    final_model.save("flips_global_model.h5")
    print("\nTraining complete. Global model saved as 'flips_global_model.h5'")

if __name__ == "__main__":
    main()
