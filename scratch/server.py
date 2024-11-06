import json
import os
import zlib
from glob import glob

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_fashionmnist_data(validation_split=0.2):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_split, random_state=42)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_models_from_selected_clients(directory, selected_clients):
    selected_clients = {client.split("/")[1] for client in selected_clients}

    return [
        tf.keras.models.load_model(os.path.join(directory, filename))
        for filename in os.listdir(directory)
        if filename in selected_clients
    ]


def get_layer_importance(json_info, layer_name):
    for layer in json_info:
        if layer[1] == layer_name:
            return layer[2]


def fedavg(models, selected_clients_json_info):

    avg_weights = []

    for layer_index in range(len(models[0].layers)):

        layer_weights = []

        for client_index, model in enumerate(models):

            layer_weight = model.layers[layer_index].get_weights()
            layer_name = models[0].layers[layer_index].name
            layer_importance = get_layer_importance(
                selected_clients_json_info[client_index]["layer_importances"], layer_name
            )

            if selected_clients_json_info[client_index]["layer_importances"][1] == layer_name:
                print(layer_name)

            layer_weights.append(layer_weight)

        if layer_weights[0]:

            averaged_layer_weights = []
            for weights_set in zip(*layer_weights):
                weights_stack = np.stack(weights_set, axis=0)
                avg_weights_set = np.mean(weights_stack, axis=0)
                averaged_layer_weights.append(avg_weights_set)

            avg_weights.append(averaged_layer_weights)
        else:

            avg_weights.append([])

    return avg_weights


def quantize_weights(weights, quantization_levels):
    quantized_weights = []

    for layer_weights in weights:
        if isinstance(layer_weights, list):

            quantized_layer = []
            for component in layer_weights:

                min_val = np.min(component)
                max_val = np.max(component)

                step_size = (max_val - min_val) / (quantization_levels - 1)

                quantized_component = np.round((component - min_val) / step_size) * step_size + min_val
                quantized_layer.append(quantized_component)

            quantized_weights.append(quantized_layer)
        else:

            min_val = np.min(layer_weights)
            max_val = np.max(layer_weights)

            step_size = (max_val - min_val) / (quantization_levels - 1)

            quantized_layer = np.round((layer_weights - min_val) / step_size) * step_size + min_val

            quantized_weights.append(quantized_layer)

    return quantized_weights


def compress_weights(weights):

    flat_weights = []

    for layer_weights in weights:
        if isinstance(layer_weights, list):

            for component in layer_weights:
                flat_weights.append(component.flatten())
        else:

            flat_weights.append(layer_weights.flatten())

    flat_weights = np.concatenate(flat_weights)

    weight_bytes = flat_weights.tobytes()

    compressed_weights = zlib.compress(weight_bytes)

    return compressed_weights


def set_model_weights(model_template, quantized_weights):
    for layer_index, layer in enumerate(model_template.layers):
        layer_weights = layer.get_weights()
        if len(layer_weights) != len(quantized_weights[layer_index]):
            raise ValueError(
                f"Layer {layer.name} expected {len(layer_weights)} weights, "
                f"but received {len(quantized_weights[layer_index])} weights."
            )

        layer.set_weights(quantized_weights[layer_index])


def load_validation_data():

    (_, _), validation_data, (_, _) = load_fashionmnist_data(validation_split=0.2)
    return validation_data


def main():
    # Define file paths and settings
    model_dir = "models/"
    clients_file = "successful_clients.json"
    output_model_path = "models/fedavg_model.keras"
    quantization_factor = 256

    # Load successful clients and their model metadata
    with open(clients_file, "r") as f:
        successful_clients = json.load(f)["successful_clients"]

    client_metadata = [json.load(open(client.replace(".keras", ".json"))) for client in successful_clients]

    # Load models and validation data
    models = load_models_from_selected_clients(model_dir, successful_clients)
    if not models:
        raise ValueError("No models found for the selected clients.")

    validation_data = load_validation_data()

    # Perform FedAvg, quantize, and compress weights
    avg_weights = fedavg(models, client_metadata)
    quantized_weights = quantize_weights(avg_weights, quantization_factor)
    compressed_weights = compress_weights(quantized_weights)

    print(f"Compressed model size: {len(compressed_weights)} bytes")

    # Save the FedAvg model with quantized weights
    fedavg_model = models[0]
    set_model_weights(fedavg_model, quantized_weights)
    fedavg_model.save(output_model_path)
    print(f"FedAvg model saved at: {output_model_path}")

    # Evaluate the FedAvg model
    val_loss, val_accuracy = fedavg_model.evaluate(*validation_data, verbose=2)

    # Replace client models with the FedAvg model
    for filename in os.listdir(model_dir):
        if filename.endswith(".keras"):
            fedavg_model.save(os.path.join(model_dir, filename))
            print(f"Replaced client model at: {filename}")

    # Calculate and log metrics
    final_compressed_size = len(compress_weights(quantize_weights(fedavg_model.get_weights(), quantization_factor)))
    with open("metrics.txt", "a+") as metrics_file:
        metrics_file.write(f"Validation Loss: {val_loss}\nValidation Accuracy: {val_accuracy}\n")

    evaluation_metrics = {
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy,
        "Compressed Model Size (bytes)": final_compressed_size,
    }
    with open("evaluation_metrics.json", "w") as json_file:
        json.dump(evaluation_metrics, json_file, indent=4)

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    print(f"Compressed Model Size: {final_compressed_size} bytes")


if __name__ == "__main__":
    main()
