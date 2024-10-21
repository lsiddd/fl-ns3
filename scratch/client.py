from fastapi import FastAPI
import shap
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
import threading
from queue import Queue
import zlib
import json
import time

app = FastAPI()

# Load data globally
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0

# Locks and queues to ensure sequential training
model_lock = threading.Lock()  # Lock to ensure sequential training
client_queue = Queue()  # Queue for managing client training requests
completed_clients = 0  # Track how many clients have finished training
expected_clients = 0  # Set by the first client to know how many are expected

class TrainRequest(BaseModel):
    n_clients: int
    client_id: int
    epochs: int
    top_n: int
    model: str  # Model filename received from the client


# Helper Functions
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
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])


def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels), verbose=2)


def process_queue():
    global completed_clients, expected_clients
    while True:
        client_request = client_queue.get()
        with model_lock:
            # Load the specific model for this client
            model = load_model_from_file(client_request.model)

            # Execute training
            train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, client_request.n_clients, client_request.client_id)
            start_time = time.time()
            train_model(model, train_images_chunk, train_labels_chunk, test_images, test_labels, epochs=client_request.epochs)
            end_time = time.time()

            # Save the updated model for this client
            model.save(client_request.model)  # Save after each client training
            completed_clients += 1  # Track completed clients

            # quantize and compress weights and calculate size
            weights = get_weights(model)
            quantized_weights = quantize_weights(weights)
            compressed_weights = compress_weights(quantized_weights)
            compressed_size = len(compressed_weights)
            quantized_model = client_request.model + ".tflite"
            print(f"Compressed model size: {compressed_size} bytes")
            
            save_weights_to_file(quantized_model, compressed_weights)

            uncompressed_size = os.path.getsize(client_request.model)

            layer_importances = rank_model_layers(model, test_images, test_labels)

            top_n_layers = [
                model.layers[layer_index]
                for layer_index, _, _ in layer_importances[: client_request.top_n]
            ]
            top_n_output_filename = f"{client_request.model.split('.')[0]}_top_{client_request.top_n}_layers.tflite"
            compressed_top_n_size = save_compressed_top_n_layers(
                model, top_n_layers, top_n_output_filename
            )

            results = {
                "uncompressed_size": uncompressed_size,
                "compressed_size": compressed_size,
                "compressed_top_n_size": compressed_top_n_size,
                "duration": (end_time - start_time) * 1000
            }

            with open(f"{client_request.model.split('.')[0]}_model_sizes.json", "w") as f:
                json.dump(results, f, indent=4)

            with open(f"{client_request.model.split('.')[0]}.finish", "w+") as finish_file:
                finish_file.write("ok")

            print(f"Client {client_request.client_id} completed training.")

            # If all clients are done, signal C++ to proceed
            if completed_clients == expected_clients:
                print("All clients have completed training. Proceed to next step.")
                # client_queue = Queue()  # Queue for managing client training requests
                completed_clients = 0  # Track how many clients have finished training
                expected_clients = 0  # Set by the first client to know how many are expected

        client_queue.task_done()  # Mark task as done



def quantize_weights(weights, quantization_levels=256):
    quantized_weights = []
    for layer_weights in weights:
        min_val = np.min(layer_weights)
        max_val = np.max(layer_weights)
        step_size = (max_val - min_val) / (quantization_levels - 1)
        quantized_layer = (np.round((layer_weights - min_val) / step_size) * step_size + min_val)
        quantized_weights.append(quantized_layer)
    return quantized_weights


def compress_weights(weights):
    flat_weights = np.concatenate([w.flatten() for w in weights])
    weight_bytes = flat_weights.tobytes()
    compressed_weights = zlib.compress(weight_bytes)
    return compressed_weights


def rank_model_layers(model, test_images, test_labels):
    base_accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
    layer_impact = []
    for i, layer in enumerate(model.layers):
        original_weights = layer.get_weights()
        if not original_weights:
            continue
        zeroed_weights = [np.zeros_like(w) for w in original_weights]
        layer.set_weights(zeroed_weights)
        perturbed_accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
        accuracy_drop = base_accuracy - perturbed_accuracy
        layer_impact.append((i, layer.name, accuracy_drop))
        layer.set_weights(original_weights)
    layer_impact.sort(key=lambda x: x[2], reverse=True)
    return layer_impact


def compute_shap_values(model, data):
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    return shap_values


def save_compressed_top_n_layers(model, top_n_layers, output_filename):
    new_model = tf.keras.models.Sequential(top_n_layers)
    new_model_weights = get_weights(new_model)
    new_quantized_weights = quantize_weights(new_model_weights)
    new_compressed_weights = compress_weights(new_quantized_weights)
    compressed_size = len(new_compressed_weights)
    return compressed_size


def save_weights_to_file(filename, compressed_weights):
    with open(filename, 'wb') as f:
        f.write(compressed_weights)


def get_weights(model: tf.keras.models.Model):
    return model.get_weights()

@app.post("/train")
def train_model_endpoint(request: TrainRequest):
    global expected_clients

    # First client sets the number of expected clients
    if completed_clients == 0:
        expected_clients = request.n_clients

    if completed_clients >= expected_clients:
        return {"message": "All clients have already completed training."}

    # Add the client training request to the queue
    client_queue.put(request)
    return {"message": f"Client {request.client_id} added to the training queue."}


# Start a background thread to process the queue
threading.Thread(target=process_queue, daemon=True).start()

# Start the service with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
