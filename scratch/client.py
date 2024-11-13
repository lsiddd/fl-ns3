import concurrent.futures
import gc
import json
import os
import time
import zlib
import traceback

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Global executor variable
executor = None


RETRY_COUNT = 3  # Number of retries for each client request
RETRY_DELAY = 5  # Delay between retries in seconds


def process_client_request_with_retries(client_request):
    """
    Wrapper function to process a client request with retry logic.
    """
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            print(f"Attempt {attempt} for client {client_request.client_id}")
            # Call the actual processing function
            process_client_request(client_request)
            print(
                f"Client {client_request.client_id} completed successfully on attempt {attempt}")
            break  # Exit loop if successful
        except Exception as e:
            print(
                f"Error processing client request {client_request.client_id} on attempt {attempt}: {e}")
            traceback.print_exc()
            if attempt == RETRY_COUNT:
                print(
                    f"Max retries reached for client {client_request.client_id}. Moving on.")
            else:
                print(
                    f"Retrying client {client_request.client_id} in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)  # Delay before retry
        finally:
            # Clean up resources to free memory
            import tensorflow as tf  # Import tensorflow inside function scope
            tf.keras.backend.clear_session()
            gc.collect()


class TrainRequest(BaseModel):
    n_clients: int
    client_id: int
    epochs: int
    top_n: int
    model: str


def process_client_request(client_request):
    # All imports inside the function to avoid issues with forking
    import os
    import gc
    import zlib
    import time
    import json
    import traceback
    import numpy as np
    import tensorflow as tf

    # Set environment variable to use CPU only
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        # Load dataset inside the function
        (train_images, train_labels), (test_images,
                                       test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = train_images[..., np.newaxis] / 255.0
        test_images = test_images[..., np.newaxis] / 255.0

        # Create test_dataset
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_images, test_labels)).batch(64).prefetch(tf.data.AUTOTUNE)

        # Define all the required functions inside the process_client_request or import them from a module
        def split_data(train_images, train_labels, n_clients, client_id, non_iid_factor=0.2):
            np.random.seed(client_id)
            unique_classes = np.unique(train_labels)
            n_classes = len(unique_classes)
            data_size = len(train_images)
            chunk_size = data_size // n_clients

            if non_iid_factor == 0:
                # IID split
                start_index = client_id * chunk_size
                end_index = data_size if client_id == n_clients - \
                    1 else (client_id + 1) * chunk_size
                return train_images[start_index:end_index], train_labels[start_index:end_index]
            else:
                # Non-IID split
                class_indices = [np.where(train_labels == c)[0]
                                 for c in unique_classes]
                n_classes_per_client = max(
                    1, int((1 - non_iid_factor) * n_classes))
                selected_classes = np.random.choice(
                    unique_classes, size=n_classes_per_client, replace=False)
                print(f"client {client_id} classes: {selected_classes}")

                client_indices = []
                for cls in selected_classes:
                    class_idx = class_indices[cls]
                    np.random.shuffle(class_idx)
                    num_samples_for_class = int(
                        (non_iid_factor + 1) * len(class_idx) / n_clients)
                    client_indices.extend(class_idx[:num_samples_for_class])

                np.random.shuffle(client_indices)
                client_data_indices = np.array_split(
                    client_indices, n_clients)[client_id]
                return train_images[client_data_indices], train_labels[client_data_indices]

        def load_model_from_file(model_filename):
            if os.path.exists(model_filename):
                print(f"Loading model from {model_filename}")
                return tf.keras.models.load_model(model_filename)
            else:
                print(
                    f"Model file {model_filename} does not exist. Creating a new model.")
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        64, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        64, (3, 3), padding="same", activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Conv2D(
                        128, (3, 3), padding="same", activation="relu"),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(
                        128, (3, 3), padding="same", activation="relu"),
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
                return model

        def get_dataset(images, labels, batch_size=64):
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.shuffle(buffer_size=10000).batch(
                batch_size).prefetch(tf.data.AUTOTUNE)
            return dataset

        def train_model(model, train_dataset, test_dataset, epochs=10):
            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
                    quantized_layer = np.round(
                        (layer_weights - min_val) / step_size) * step_size + min_val
                quantized_weights.append(quantized_layer.astype(np.float32))
            return quantized_weights

        def prune_weights(weights, pruning_percentage=0.2):
            pruned_weights = []
            for layer_weights in weights:
                if layer_weights.ndim > 0:
                    threshold = np.percentile(
                        np.abs(layer_weights), pruning_percentage * 100)
                    pruned_layer_weights = np.where(
                        np.abs(layer_weights) < threshold, 0, layer_weights)
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

        def get_weights(model: tf.keras.models.Model):
            return model.get_weights()

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

        def save_weights_to_file(filename, compressed_weights_list):
            with open(filename, "wb") as f:
                for compressed_weights in compressed_weights_list:
                    f.write(compressed_weights)

        def rank_model_layers(model, test_dataset, subset_size=1000, batch_size=64, max_workers=4):
            # Use a subset of the test dataset
            test_dataset_subset = test_dataset.take(subset_size // batch_size)
            base_accuracy = model.evaluate(test_dataset_subset, verbose=0)[1]

            def evaluate_layer_impact(i, layer):
                original_weights = layer.get_weights()
                if not original_weights:
                    return i, layer.name, 0

                zeroed_weights = [np.zeros_like(w) for w in original_weights]
                layer.set_weights(zeroed_weights)
                perturbed_accuracy = model.evaluate(
                    test_dataset_subset, verbose=0)[1]
                accuracy_drop = base_accuracy - perturbed_accuracy
                layer.set_weights(original_weights)

                del original_weights
                del zeroed_weights

                return i, layer.name, accuracy_drop

            layer_impact = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(
                    evaluate_layer_impact, i, layer) for i, layer in enumerate(model.layers)]
                for future in concurrent.futures.as_completed(futures):
                    layer_impact.append(future.result())

            layer_impact.sort(key=lambda x: x[2], reverse=True)
            return layer_impact

        # Now proceed with the training process
        model = load_model_from_file(client_request.model)

        train_images_chunk, train_labels_chunk = split_data(
            train_images, train_labels, client_request.n_clients, client_request.client_id
        )

        train_dataset = get_dataset(train_images_chunk, train_labels_chunk)
        start_time = time.time()
        history = train_model(
            model,
            train_dataset,
            test_dataset,
            epochs=client_request.epochs,
        )
        end_time = time.time()

        model.save(client_request.model)

        weights = get_weights(model)
        quantized_weights = quantize_weights(weights)
        pruned_weights = prune_weights(quantized_weights)
        compressed_weights = compress_weights(pruned_weights)

        compressed_size = sum(len(cw) for cw in compressed_weights)
        quantized_model = client_request.model + ".tflite"
        print(f"Compressed model size: {compressed_size} bytes")

        save_weights_to_file(quantized_model, quantized_weights)

        uncompressed_size = os.path.getsize(client_request.model)

        time_to_rank_start = time.time()
        layer_importances = rank_model_layers(
            model, test_dataset, batch_size=64)
        time_to_rank_end = time.time()
        print(
            f"Ranking layers took {time_to_rank_end - time_to_rank_start} seconds")

        top_n_layers = [
            model.layers[layer_index] for layer_index, _, _ in layer_importances[: client_request.top_n]
        ]
        top_n_output_filename = f"{client_request.model.split('.')[0]}_top_{client_request.top_n}_layers.tflite"
        
        compressed_top_n_size = save_compressed_top_n_layers(
            model, top_n_layers, top_n_output_filename)

        results = {
            "uncompressed_size": uncompressed_size,
            "compressed_size": compressed_size,
            "number_of_samples": len(train_images_chunk),
            "compressed_top_n_size": compressed_top_n_size,
            "duration": (end_time - start_time) * 1000,
            "loss": history.history["loss"][-1] if history else None,
            "accuracy": history.history["accuracy"][-1] if history else None,
            "val_loss": history.history["val_loss"][-1] if history else None,
            "val_accuracy": history.history["val_accuracy"][-1] if history else None,
            "layer_importances": layer_importances,
        }

        with open(f"{client_request.model.split('.')[0]}.json", "w") as f:
            json.dump(results, f, indent=4)

        with open(f"{client_request.model.split('.')[0]}.finish", "w+") as finish_file:
            finish_file.write("ok")

        print(f"Client {client_request.client_id} completed training.")

    except Exception as e:
        print(
            f"Error processing client request {client_request.client_id}: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources to free memory
        tf.keras.backend.clear_session()
        gc.collect()


@app.post("/train")
def train_model_endpoint(request: TrainRequest):
    print(f"Received training request from client {request.client_id}")
    # Submit the task with retry wrapper to the executor
    future = executor.submit(process_client_request_with_retries, request)
    return {"message": f"Client {request.client_id} training started.\n"}


if __name__ == "__main__":
    import uvicorn
    import multiprocessing

    # Set the multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # Initialize the executor after setting the start method
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)

    uvicorn.run(app, host="0.0.0.0", port=8182)
