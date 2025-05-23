import gc
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional


def split_data(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    n_clients: int,
    client_id: int,
    alpha: float = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(client_id)
    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)

    # Get the indices of each class
    class_indices = {cls: np.where(train_labels == cls)[0] for cls in unique_classes}

    # Use Dirichlet distribution to generate proportions for this client
    class_proportions = np.random.dirichlet(alpha * np.ones(n_clients), n_classes)

    # Select data for the specific client_id
    client_indices = []
    for cls in unique_classes:
        cls_indices = class_indices[cls]
        np.random.shuffle(cls_indices)

        # Allocate data to the client based on the proportion
        n_samples_for_client = int(len(cls_indices) * class_proportions[cls, client_id])
        client_indices.extend(cls_indices[:n_samples_for_client])

    np.random.shuffle(client_indices)

    # Return the data for this client
    return train_images[client_indices], train_labels[client_indices]


def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)
            ),
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
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def get_dataset(
    images: np.ndarray, labels: np.ndarray, batch_size: int = 64
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = (
        dataset.shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    epochs: int = 1,
) -> tf.keras.callbacks.History:
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2)


def get_weights(model: tf.keras.Model) -> List[np.ndarray]:
    return model.get_weights()


def set_weights(model: tf.keras.Model, weights: List[np.ndarray]) -> None:
    model.set_weights(weights)


def average_weights(
    weights_list: List[List[np.ndarray]], weights_scaling_factors: Optional[List[float]] = None
) -> List[np.ndarray]:
    if weights_scaling_factors is None:
        weights_scaling_factors = [1.0 / len(weights_list)] * len(weights_list)
    avg_weights = []
    for weights in zip(*weights_list):
        weighted_sum = sum(w * s for w, s in zip(weights, weights_scaling_factors))
        avg_weights.append(weighted_sum)
    return avg_weights


def client_update(
    client_id: int,
    global_weights: List[np.ndarray],
    global_trainable_weights: List[np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_dataset: tf.data.Dataset,
    n_clients: int,
    epochs: int,
    mu: float,
) -> Tuple[List[np.ndarray], int]:
    # Each client starts with the global model
    client_model = create_model()
    set_weights(client_model, global_weights)

    # Split data for client
    train_images_chunk, train_labels_chunk = split_data(
        train_images, train_labels, n_clients, client_id
    )
    client_samples = len(train_images_chunk)

    # Prepare train_dataset
    train_dataset = get_dataset(train_images_chunk, train_labels_chunk)

    # Compile model with FedProx optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = client_model(images, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            loss = tf.reduce_mean(loss)
            # Proximal term
            proximal_term = 0.0
            for w, gw in zip(client_model.trainable_variables, global_trainable_weights):
                proximal_term += tf.nn.l2_loss(w - gw)
            loss += (mu / 2) * proximal_term
        grads = tape.gradient(loss, client_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, client_model.trainable_variables))
        return loss

    # Custom training loop
    for epoch in range(epochs):
        for images, labels in train_dataset:
            train_step(images, labels)

    local_weights = client_model.get_weights()

    # Clean up
    tf.keras.backend.clear_session()
    gc.collect()

    return local_weights, client_samples


def aggregate_models(
    client_weights: List[List[np.ndarray]], client_samples: List[int]
) -> List[np.ndarray]:
    # Weight by number of samples
    total_samples = sum(client_samples)
    scaling_factors = [num / total_samples for num in client_samples]
    avg_weights = average_weights(client_weights, scaling_factors)
    return avg_weights


def main():
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images[..., np.newaxis] / 255.0
    test_images = test_images[..., np.newaxis] / 255.0

    # Prepare test dataset
    test_dataset = get_dataset(test_images, test_labels)

    n_clients = 4
    n_rounds = 10
    epochs = 1
    mu = 0.01  # Proximal term coefficient for FedProx

    # Initialize global model
    global_model = create_model()
    global_weights = global_model.get_weights()
    global_model.set_weights(global_weights)
    global_trainable_weights = [w.numpy() for w in global_model.trainable_variables]

    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num+1} ---")
        client_weights = []
        client_samples = []

        for client_id in range(n_clients):
            print(f"\nClient {client_id+1}/{n_clients}")
            local_weights, samples = client_update(
                client_id,
                global_weights,
                global_trainable_weights,
                train_images,
                train_labels,
                test_dataset,
                n_clients,
                epochs,
                mu,
            )
            client_weights.append(local_weights)
            client_samples.append(samples)

        # Aggregate client models
        print("\nAggregating client models")
        global_weights = aggregate_models(client_weights, client_samples)

        # Update global model
        global_model.set_weights(global_weights)
        global_trainable_weights = [w.numpy() for w in global_model.trainable_variables]

        # Evaluate global model
        global_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        loss, accuracy = global_model.evaluate(test_dataset, verbose=0)
        print(f"Round {round_num+1} Global model accuracy: {accuracy:.4f}")

    # Save final global model
    global_model.save("fedprox_global_model.h5")
    print("\nTraining complete. Global model saved as 'fedprox_global_model.h5'")


if __name__ == "__main__":
    main()
