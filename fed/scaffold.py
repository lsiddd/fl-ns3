import gc
import tensorflow as tf
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import traceback

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
    model: tf.keras.Model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

def get_dataset(images: np.ndarray, labels: np.ndarray, batch_size: int = 64) -> tf.data.Dataset:
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

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
        # Should not reach here in SCAFFOLD-only code
        raise ValueError("Invalid aggregation method")

    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()

    return local_weights, client_samples, new_client_cv

def aggregate_models(client_weights: List[List[np.ndarray]], client_samples: List[int], global_weights: List[np.ndarray], aggregation_method: str, global_model: tf.keras.Model = None, **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if aggregation_method == "SCAFFOLD":
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

def main() -> None:
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images: np.ndarray = train_images[..., np.newaxis] / 255.0
    test_images: np.ndarray = test_images[..., np.newaxis] / 255.0

    # Create test_dataset
    test_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64).prefetch(tf.data.AUTOTUNE)

    n_clients: int = 4
    n_rounds: int = 50
    epochs: int = 1  # Local epochs per round
    aggregation_method: str = "SCAFFOLD"

    # Initialize global model
    global_model: tf.keras.Model = create_model()
    global_weights: List[np.ndarray] = global_model.get_weights()
    trainable_variables: List[tf.Variable] = global_model.trainable_variables

    # Initialize control variates for SCAFFOLD
    server_control_variate: List[np.ndarray] = [np.zeros_like(v) for v in trainable_variables]
    client_control_variates: List[List[np.ndarray]] = [[np.zeros_like(v) for v in trainable_variables] for _ in range(n_clients)]

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
                local_weights, samples, new_client_cv = client_update(
                    client_id, global_weights, train_images, train_labels, test_dataset,
                    aggregation_method, n_clients=n_clients,
                    client_control_variates=client_control_variates,
                    server_control_variate=server_control_variate,
                    epochs=epochs
                )
                client_weights.append(local_weights)
                client_samples.append(samples)
                if new_client_cv is not None:
                    client_control_variates[client_id] = new_client_cv
            except Exception as e:
                print(f"Error processing client {client_id}: {e}")
                traceback.print_exc()

        # Aggregate client models
        print("\nAggregating client models")
        global_weights, server_control_variate = aggregate_models(
            client_weights, client_samples, global_weights, aggregation_method,
            global_model=global_model,
            client_control_variates=client_control_variates,
            server_control_variate=server_control_variate
        )

        # Update global model
        global_model.set_weights(global_weights)

        # Evaluate global model
        global_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loss: float
        accuracy: float
        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Round {round_num+1} Global model accuracy: {accuracy:.4f}")

    # Save final global model
    global_model.save("scaffold_global_model.h5")
    print("\nTraining complete. Global model saved as 'scaffold_global_model.h5'")

if __name__ == "__main__":
    main()
