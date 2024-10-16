import json
from copy import deepcopy
import os
import shap
import argparse
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def split_data(train_images, train_labels, n_clients, client_id):
    assert 0 <= client_id < n_clients, "client_id should be between 0 and n_clients-1"

    data_size = len(train_images)
    chunk_size = data_size // n_clients
    start_index = client_id * chunk_size
    end_index = (client_id + 1) * chunk_size if client_id < n_clients - 1 else data_size

    train_images_chunk = train_images[start_index:end_index]
    train_labels_chunk = train_labels[start_index:end_index]

    return train_images_chunk, train_labels_chunk


def load_and_preprocess_data():
    from tensorflow.keras import datasets
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def create_cnn_model():
    from tensorflow.keras import models, layers
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


def train_model(model, train_images, train_labels, test_images, test_labels, epochs=10):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

    return history


def save_model(model, filename='fashionmnist_model.h5'):
    model.save(filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    from tensorflow.keras import models
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        return models.load_model(filename)
    else:
        print(f"Model file {filename} does not exist. Exiting.")
        return create_cnn_model()


def compress_and_quantize_model(model, output_filename):
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_filename, 'wb') as f:
        f.write(tflite_model)

    model_size = os.path.getsize(output_filename)
    print(f"Compressed and quantized model saved to {output_filename}")
    print(f"Compressed model size: {model_size} bytes")

    return model_size


def rank_model_layers(model, test_images, test_labels):
    base_accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
    print(f"Base accuracy of the original model: {base_accuracy:.4f}")

    layer_impact = []

    for i, layer in enumerate(model.layers):
        temp_model = deepcopy(model)

        temp_model.layers[i].trainable = False
        temp_model.layers[i].set_weights([np.zeros_like(w) for w in temp_model.layers[i].get_weights()])

        temp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        perturbed_accuracy = temp_model.evaluate(test_images, test_labels, verbose=0)[1]

        accuracy_drop = base_accuracy - perturbed_accuracy
        layer_impact.append((i, layer.name, accuracy_drop))
        print(f"Layer {i} ({layer.name}): Perturbed accuracy = {perturbed_accuracy:.4f}, Accuracy drop = {accuracy_drop:.4f}")

    layer_impact.sort(key=lambda x: x[2], reverse=True)

    print("\nLayer ranking by importance:")
    for i, (layer_index, layer_name, accuracy_drop) in enumerate(layer_impact):
        print(f"{i+1}. Layer {layer_index} ({layer_name}): Accuracy drop = {accuracy_drop:.4f}")

    return layer_impact


def compute_shap_values(model, data, model_filename):
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)

    layer_importances = []
    for layer_idx, layer in enumerate(model.layers):
        if not layer.trainable:
            continue
        shap_value_layer = shap_values[layer_idx]
        shap_score = np.mean(np.abs(shap_value_layer))
        layer_importances.append({"layer_idx": layer_idx, "layer_name": layer.name, "shap_score": shap_score})
        print(f"Layer {layer_idx} ({layer.name}) - SHAP score: {shap_score:.4f}")

    layer_importances.sort(key=lambda x: x['shap_score'], reverse=True)

    print("\nLayer ranking by SHAP values importance:")
    for i, layer_data in enumerate(layer_importances):
        print(f"{i+1}. Layer {layer_data['layer_idx']} ({layer_data['layer_name']}): SHAP score = {layer_data['shap_score']:.4f}")

    shap_json_filename = model_filename.split(".")[0] + "_shap_importances.json"
    with open(shap_json_filename, 'w') as json_file:
        json.dump(layer_importances, json_file, indent=4)

    print(f"SHAP layer importances saved to {shap_json_filename}")

    return layer_importances


def save_compressed_top_n_layers(model, top_n_layers, output_filename):
    import tensorflow as tf

    new_model = tf.keras.models.Sequential()

    for layer_idx, layer in enumerate(top_n_layers):
        new_model.add(model.layers[layer_idx])

    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_filename, 'wb') as f:
        f.write(tflite_model)

    model_size = os.path.getsize(output_filename)
    print(f"Compressed and quantized top {len(top_n_layers)} layers saved to {output_filename}")
    print(f"Compressed top N layers model size: {model_size} bytes")

    return model_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="fashionmnist_quantized_model.keras")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_clients', type=int, required=True)
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--top_n', type=int, default=3, help='Number of top layers to save after SHAP ranking')
    args = parser.parse_args()

    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    train_images_chunk, train_labels_chunk = split_data(train_images, train_labels, args.n_clients, args.id)

    if args.model:
        model = load_model(args.model)
    else:
        model = create_cnn_model()

    quantized_model = args.model.split(".")[0] + ".tflite"

    train_model(model, train_images_chunk, train_labels_chunk, test_images, test_labels, epochs=args.epochs)
    save_model(model, args.model)

    uncompressed_size = os.path.getsize(args.model)
    print(f"Uncompressed model size: {uncompressed_size} bytes")

    compressed_size = compress_and_quantize_model(model, quantized_model)

    data_sample = test_images[:100]
    shap_values = compute_shap_values(model, data_sample, args.model)

    top_n_layers = shap_values[:args.top_n]
    top_n_output_filename = f"{args.model.split('.')[0]}_top_{args.top_n}_layers.tflite"
    compressed_top_n_size = save_compressed_top_n_layers(model, top_n_layers, top_n_output_filename)

    results = {
        "uncompressed_size": uncompressed_size,
        "compressed_size": compressed_size,
        "compressed_top_n_size": compressed_top_n_size
    }

    with open(f"{args.model.split('.')[0]}_model_sizes.json", 'w') as f:
        json.dump(results, f, indent=4)

    print("Model sizes saved to JSON file.")
