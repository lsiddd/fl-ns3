import concurrent.futures
import gc
import json
import os
import threading
import time
import zlib
from queue import Queue

import numpy as np
import shap
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:

        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPUs are available and memory growth is enabled: {physical_devices}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU available, using CPU.")


app = FastAPI()


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0


model_lock = threading.Lock()
client_queue = Queue()
completed_clients = 0
expected_clients = 0


class TrainRequest(BaseModel):
    n_clients: int
    client_id: int
    epochs: int
    top_n: int
    model: str


def split_data(train_images, train_labels, n_clients, client_id):
    data_size = len(train_images)
    chunk_size = data_size // n_clients
    start_index = client_id * chunk_size
    end_index = data_size if client_id == n_clients - 1 else (client_id + 1) * chunk_size
    return train_images[start_index:end_index], train_labels[start_index:end_index]


def load_model_from_file(model_filename):
    if os.path.exists(model_filename):
        print(f"Loading model from {model_filename}")
        return tf.keras.models.load_model(model_filename)
    else:
        print(f"Model file {model_filename} does not exist. Creating a new model.")
        return tf.keras.Sequential(
            [
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
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )


def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels), verbose=2)


def quantize_weights(weights, quantization_levels=256):
    quantized_weights = []
    for layer_weights in weights:
        min_val = np.min(layer_weights)
        max_val = np.max(layer_weights)
        step_size = (max_val - min_val) / (quantization_levels - 1)
        quantized_layer = np.round((layer_weights - min_val) / step_size) * step_size + min_val
        quantized_weights.append(quantized_layer)
    return quantized_weights


def prune_weights(weights, pruning_percentage=0.2):
    """Prune the weights by setting a percentage of the smallest weights to zero."""
    pruned_weights = []
    for layer_weights in weights:
        if layer_weights.ndim > 0:  # If it's not an empty layer
            threshold = np.percentile(np.abs(layer_weights), pruning_percentage * 100)
            pruned_layer_weights = np.where(np.abs(layer_weights) < threshold, 0, layer_weights)
            pruned_weights.append(pruned_layer_weights)
        else:
            pruned_weights.append(layer_weights)  # No need to prune empty weights
    return pruned_weights


def compress_weights(weights):
    compressed_weights = []
    for layer_weights in weights:
        weight_bytes = layer_weights.tobytes()
        compressed_layer_weights = zlib.compress(weight_bytes)
        compressed_weights.append(compressed_layer_weights)
    return compressed_weights


def rank_model_layers(model, test_images, test_labels, subset_size=1000, max_workers=4):
    if subset_size and subset_size < len(test_images):
        indices = np.random.choice(len(test_images), subset_size, replace=False)
        test_images_subset = test_images[indices]
        test_labels_subset = test_labels[indices]
    else:
        test_images_subset = test_images
        test_labels_subset = test_labels

    base_accuracy = model.evaluate(test_images_subset, test_labels_subset, verbose=0)[1]

    def evaluate_layer_impact(i, layer):
        """Helper function to evaluate the accuracy drop when a layer's weights are zeroed out."""
        original_weights = layer.get_weights()
        if not original_weights:

            return i, layer.name, 0

        zeroed_weights = [np.zeros_like(w) for w in original_weights]
        layer.set_weights(zeroed_weights)

        perturbed_accuracy = model.evaluate(test_images_subset, test_labels_subset, verbose=0)[1]
        accuracy_drop = base_accuracy - perturbed_accuracy

        layer.set_weights(original_weights)

        del original_weights
        del zeroed_weights

        return i, layer.name, accuracy_drop

    layer_impact = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_layer_impact, i, layer) for i, layer in enumerate(model.layers)]
        for future in concurrent.futures.as_completed(futures):
            layer_impact.append(future.result())

    layer_impact.sort(key=lambda x: x[2], reverse=True)

    del test_images_subset
    del test_labels_subset

    return layer_impact


def get_weights(model: tf.keras.models.Model):
    return model.get_weights()


def save_compressed_top_n_layers(model, top_n_layers, output_filename):
    new_model = tf.keras.models.Sequential(top_n_layers)
    new_model_weights = get_weights(new_model)
    new_quantized_weights = quantize_weights(new_model_weights)
    new_compressed_weights = compress_weights(new_quantized_weights)
    compressed_size = sum(len(cw) for cw in new_compressed_weights)
    return compressed_size


def save_weights_to_file(filename, compressed_weights_list):
    with open(filename, "wb") as f:
        for compressed_weights in compressed_weights_list:
            f.write(compressed_weights)


def process_queue():
    global completed_clients, expected_clients
    while True:
        client_request = client_queue.get()
        with model_lock:

            model = load_model_from_file(client_request.model)
            try:

                train_images_chunk, train_labels_chunk = split_data(
                    train_images, train_labels, client_request.n_clients, client_request.client_id
                )
                start_time = time.time()
                history = train_model(
                    model,
                    train_images_chunk,
                    train_labels_chunk,
                    test_images,
                    test_labels,
                    epochs=client_request.epochs,
                )
                end_time = time.time()

                model.save(client_request.model)
                completed_clients += 1

                weights = get_weights(model)
                quantized_weights = quantize_weights(weights)
                pruned_weights = prune_weights(quantized_weights)
                compressed_weights = compress_weights(pruned_weights)

                compressed_size = sum(len(cw) for cw in compressed_weights)
                quantized_model = client_request.model + ".tflite"
                print(f"Compressed model size: {compressed_size} bytes")

                save_weights_to_file(quantized_model, compressed_weights)

                uncompressed_size = os.path.getsize(client_request.model)

                time_to_rank_start = time.time()
                layer_importances = rank_model_layers(model, test_images, test_labels)
                time_to_rank_end = time.time()
                print(f"Ranking layers took {time_to_rank_end - time_to_rank_start} seconds")

                top_n_layers = [
                    model.layers[layer_index] for layer_index, _, _ in layer_importances[: client_request.top_n]
                ]
                top_n_output_filename = f"{client_request.model.split('.')[0]}_top_{client_request.top_n}_layers.tflite"
                compressed_top_n_size = save_compressed_top_n_layers(model, top_n_layers, top_n_output_filename)

                results = {
                    "uncompressed_size": uncompressed_size,
                    "compressed_size": compressed_size,
                    "number_of_samples": len(train_images_chunk),
                    "compressed_top_n_size": compressed_top_n_size,
                    "duration": (end_time - start_time) * 1000,
                    "loss": history.history["loss"][0],
                    "accuracy": history.history["accuracy"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_accuracy"][0],
                    "layer_importances": layer_importances,
                }

                with open(f"{client_request.model.split('.')[0]}.json", "w") as f:
                    json.dump(results, f, indent=4)

                with open(f"{client_request.model.split('.')[0]}.finish", "w+") as finish_file:
                    finish_file.write("ok")

                print(f"Client {client_request.client_id} completed training.")
            finally:

                del model
                del train_images_chunk
                del train_labels_chunk
                del history
                del weights
                del quantized_weights
                del compressed_weights
                del layer_importances
                del top_n_layers
                tf.keras.backend.clear_session()
                gc.collect()

                if completed_clients == expected_clients:
                    print("All clients have completed training. Proceed to next step.")
                    completed_clients = 0
                    expected_clients = 0

            client_queue.task_done()


def compute_shap_values(model, data):
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values


@app.post("/train")
def train_model_endpoint(request: TrainRequest):
    global expected_clients

    if completed_clients == 0:
        expected_clients = request.n_clients

    if completed_clients >= expected_clients:
        return {"message": "All clients have already completed training."}

    client_queue.put(request)
    return {"message": f"Client {request.client_id} added to the training queue.\n"}


threading.Thread(target=process_queue, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8182)
