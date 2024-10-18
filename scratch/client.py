import json
import zlib
import os
import argparse
import numpy as np
import tensorflow as tf

# os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def split_data(train_images, train_labels, n_clients, client_id):
    data_size = len(train_images)
    chunk_size = data_size // n_clients
    start_index = client_id * chunk_size
    end_index = (
        data_size if client_id == n_clients - 1 else (client_id + 1) * chunk_size
    )
    return train_images[start_index:end_index], train_labels[start_index:end_index]


def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    train_images = train_images[..., np.newaxis] / 255.0
    test_images = test_images[..., np.newaxis] / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def create_cnn_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)
            ),
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
        ]
    )
    return model


def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        verbose=2,
    )


def save_model(model, filename="fashionmnist_model.keras"):
    model.save(filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        return tf.keras.models.load_model(filename)
    print(f"Model file {filename} does not exist. Exiting.")
    return create_cnn_model()


# Function to quantize weights
def quantize_weights(weights, quantization_levels=256):
    quantized_weights = []
    for layer_weights in weights:
        # Get the min and max of the weights in each layer
        min_val = np.min(layer_weights)
        max_val = np.max(layer_weights)

        # Compute the step size for quantization
        step_size = (max_val - min_val) / (quantization_levels - 1)

        # Quantize the weights
        quantized_layer = (
            np.round((layer_weights - min_val) / step_size) * step_size + min_val
        )

        # Append the quantized weights
        quantized_weights.append(quantized_layer)

    return quantized_weights


def compress_weights(weights):
    # Flatten the weights to compress
    flat_weights = np.concatenate([w.flatten() for w in weights])

    # Convert to bytes
    weight_bytes = flat_weights.tobytes()

    # Compress using zlib
    compressed_weights = zlib.compress(weight_bytes)

    return compressed_weights

def save_weights_to_file(filename, compressed_weights):
    with open(filename, 'wb') as f:
        f.write(compressed_weights)

def rank_model_layers(model, test_images, test_labels):
    base_accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
    print(f"Base accuracy of the original model: {base_accuracy:.4f}")

    layer_impact = []

    for i, layer in enumerate(model.layers):
        original_weights = layer.get_weights()
        if not original_weights:  # Skip layers without weights
            continue

        zeroed_weights = [np.zeros_like(w) for w in original_weights]
        layer.set_weights(zeroed_weights)
        perturbed_accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
        accuracy_drop = base_accuracy - perturbed_accuracy
        layer_impact.append((i, layer.name, accuracy_drop))

        # Restore original weights
        layer.set_weights(original_weights)

        print(
            f"Layer {i} ({layer.name}): Perturbed accuracy = {perturbed_accuracy:.4f}, Accuracy drop = {accuracy_drop:.4f}"
        )

    layer_impact.sort(key=lambda x: x[2], reverse=True)

    print("\nLayer ranking by importance:")
    for i, (layer_index, layer_name, accuracy_drop) in enumerate(layer_impact):
        print(
            f"{i+1}. Layer {layer_index} ({layer_name}): Accuracy drop = {accuracy_drop:.4f}"
        )

    return layer_impact


def save_compressed_top_n_layers(model, top_n_layers, output_filename):
    new_model = tf.keras.models.Sequential(top_n_layers)
    new_model_weights = get_weights(new_model)
    new_quantized_weights = quantize_weights(new_model_weights)
    new_compressed_weights = compress_weights(new_quantized_weights)
    compressed_size = len(new_compressed_weights)
    
    print(
        f"Compressed top {len(top_n_layers)} layers saved to {output_filename}, size: {compressed_size} bytes"
    )
    return compressed_size


def compute_shap_values(model, data, model_filename):
    import shap

    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)

    layer_importances = []
    for layer_idx, layer in enumerate(model.layers):
        if not layer.trainable:
            continue
        shap_value_layer = shap_values[layer_idx]
        shap_score = np.mean(np.abs(shap_value_layer))
        layer_importances.append(
            {"layer_idx": layer_idx, "layer_name": layer.name, "shap_score": shap_score}
        )
        print(f"Layer {layer_idx} ({layer.name}) - SHAP score: {shap_score:.4f}")

    layer_importances.sort(key=lambda x: x["shap_score"], reverse=True)

    print("\nLayer ranking by SHAP values importance:")
    for i, layer_data in enumerate(layer_importances):
        print(
            f"{i+1}. Layer {layer_data['layer_idx']} ({layer_data['layer_name']}): SHAP score = {layer_data['shap_score']:.4f}"
        )

    shap_json_filename = model_filename.split(".")[0] + "_shap_importances.json"
    with open(shap_json_filename, "w") as json_file:
        json.dump(layer_importances, json_file, indent=4)

    print(f"SHAP layer importances saved to {shap_json_filename}")

    return layer_importances


def get_weights(model: tf.keras.models.Model):
    return model.get_weights()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fashionmnist_quantized_model.h5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_clients", type=int, required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="Number of top layers to save after ranking",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    model = load_model(args.model)

    # load data and train model
    (train_images, train_labels), (test_images, test_labels) = (
        load_and_preprocess_data()
    )
    train_images_chunk, train_labels_chunk = split_data(
        train_images, train_labels, args.n_clients, args.id
    )
    quantized_model = args.model.split(".")[0] + ".tflite"
    train_model(
        model,
        train_images_chunk,
        train_labels_chunk,
        test_images,
        test_labels,
        epochs=args.epochs,
    )
    save_model(model, args.model)

    # quantize and compress weights and calculate size
    weights = get_weights(model)
    quantized_weights = quantize_weights(weights)
    compressed_weights = compress_weights(quantized_weights)
    compressed_size = len(compressed_weights)
    print(f"Compressed model size: {compressed_size} bytes")
    
    save_weights_to_file(quantized_model, compressed_weights)

    uncompressed_size = os.path.getsize(args.model)

    layer_importances = rank_model_layers(model, test_images, test_labels)

    top_n_layers = [
        model.layers[layer_index]
        for layer_index, _, _ in layer_importances[: args.top_n]
    ]
    top_n_output_filename = f"{args.model.split('.')[0]}_top_{args.top_n}_layers.tflite"
    compressed_top_n_size = save_compressed_top_n_layers(
        model, top_n_layers, top_n_output_filename
    )

    results = {
        "uncompressed_size": uncompressed_size,
        "compressed_size": compressed_size,
        "compressed_top_n_size": compressed_top_n_size,
    }

    with open(f"{args.model.split('.')[0]}_model_sizes.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Model sizes saved to JSON file.")
