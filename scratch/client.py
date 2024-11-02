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
import concurrent.futures
import time
import gc  # Added for garbage collection

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth to prevent TensorFlow from occupying all the GPU memory
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPUs are available and memory growth is enabled: {physical_devices}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU available, using CPU.")


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
            try:
                # Execute training
                train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, client_request.n_clients, client_request.client_id)
                start_time = time.time()
                history = train_model(model, train_images_chunk, train_labels_chunk, test_images, test_labels, epochs=client_request.epochs)
                end_time = time.time()

                # Save the updated model for this client
                model.save(client_request.model)  # Save after each client training
                completed_clients += 1  # Track completed clients

                # Quantize and compress weights and calculate size
                weights = get_weights(model)
                quantized_weights = quantize_weights(weights)
                compressed_weights = compress_weights(quantized_weights)
                compressed_size = sum(len(cw) for cw in compressed_weights)  # Adjusted for list of compressed weights
                quantized_model = client_request.model + ".tflite"
                print(f"Compressed model size: {compressed_size} bytes")
                
                save_weights_to_file(quantized_model, compressed_weights)

                uncompressed_size = os.path.getsize(client_request.model)

                time_to_rank_start = time.time()
                layer_importances = rank_model_layers(model, test_images, test_labels)
                time_to_rank_end = time.time()
                print(f"Ranking layers took {time_to_rank_end - time_to_rank_start} seconds")

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
                    "number_of_samples": len(train_images_chunk),
                    "compressed_top_n_size": compressed_top_n_size,
                    "duration": (end_time - start_time) * 1000,
                    "loss": history.history['loss'][0],
                    "accuracy": history.history['accuracy'][0],
                    "val_loss": history.history['val_loss'][0],
                    "val_accuracy": history.history['val_accuracy'][0],
                    "layer_importances": layer_importances,
                }

                with open(f"{client_request.model.split('.')[0]}_model_sizes.json", "w") as f:
                    json.dump(results, f, indent=4)

                with open(f"{client_request.model.split('.')[0]}.finish", "w+") as finish_file:
                    finish_file.write("ok")

                print(f"Client {client_request.client_id} completed training.")
            finally:
                # Free up memory
                del model
                del train_images_chunk
                del train_labels_chunk
                del history
                del weights
                del quantized_weights
                del compressed_weights
                del layer_importances
                del top_n_layers
                tf.keras.backend.clear_session()  # Clear Keras backend session
                gc.collect()  # Force garbage collection

                # If all clients are done, reset counters
                if completed_clients == expected_clients:
                    print("All clients have completed training. Proceed to next step.")
                    completed_clients = 0
                    expected_clients = 0

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
    compressed_weights = []
    for layer_weights in weights:
        weight_bytes = layer_weights.tobytes()
        compressed_layer_weights = zlib.compress(weight_bytes)
        compressed_weights.append(compressed_layer_weights)
    return compressed_weights


def rank_model_layers(model, test_images, test_labels, subset_size=1000, max_workers=4):
    """
    Rank the importance of layers by measuring the accuracy drop when layer weights are zeroed out.
    Uses parallel processing to evaluate layer impacts concurrently and samples a subset of test data to speed up evaluation.
    
    Parameters:
        model: Trained Keras model.
        test_images: Test images dataset.
        test_labels: Test labels corresponding to test_images.
        subset_size: Number of samples to use for each evaluation (to speed up evaluation).
        max_workers: Number of parallel workers to use for evaluating layer impacts.

    Returns:
        List of tuples containing layer index, layer name, and accuracy drop.
    """
    # Sample a subset of the test set for faster evaluation
    if subset_size and subset_size < len(test_images):
        indices = np.random.choice(len(test_images), subset_size, replace=False)
        test_images_subset = test_images[indices]
        test_labels_subset = test_labels[indices]
    else:
        test_images_subset = test_images
        test_labels_subset = test_labels
    
    # Get base accuracy of the model on the subset
    base_accuracy = model.evaluate(test_images_subset, test_labels_subset, verbose=0)[1]
    
    def evaluate_layer_impact(i, layer):
        """Helper function to evaluate the accuracy drop when a layer's weights are zeroed out."""
        original_weights = layer.get_weights()
        if not original_weights:
            return i, layer.name, 0  # Skip layers without weights (like activation or dropout)
        
        # Zero out the layer's weights
        zeroed_weights = [np.zeros_like(w) for w in original_weights]
        layer.set_weights(zeroed_weights)
        
        # Evaluate the model with zeroed weights
        perturbed_accuracy = model.evaluate(test_images_subset, test_labels_subset, verbose=0)[1]
        accuracy_drop = base_accuracy - perturbed_accuracy
        
        # Restore the original weights
        layer.set_weights(original_weights)

        # Free up memory
        del original_weights
        del zeroed_weights
        
        return i, layer.name, accuracy_drop

    # Use concurrent processing to evaluate layers in parallel
    layer_impact = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_layer_impact, i, layer) for i, layer in enumerate(model.layers)]
        for future in concurrent.futures.as_completed(futures):
            layer_impact.append(future.result())
    
    # Sort layers by the accuracy drop in descending order
    layer_impact.sort(key=lambda x: x[2], reverse=True)

    # Free up memory
    del test_images_subset
    del test_labels_subset
    
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
    compressed_size = sum(len(cw) for cw in new_compressed_weights)
    return compressed_size


def save_weights_to_file(filename, compressed_weights_list):
    with open(filename, 'wb') as f:
        for compressed_weights in compressed_weights_list:
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
    return {"message": f"Client {request.client_id} added to the training queue.\n"}


# Start a background thread to process the queue
threading.Thread(target=process_queue, daemon=True).start()

# Start the service with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8182)
