import json
import os
import zlib
import argparse
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


def evaluate_layer_importance(model, validation_data):
    """
    Zero out each layer individually and measure the drop in validation accuracy.
    Returns a dictionary of layers with their corresponding impact scores.
    """
    layer_importance = {}
    original_weights = [layer.get_weights() for layer in model.layers]
    _, original_accuracy = model.evaluate(*validation_data, verbose=0)

    for i, layer in enumerate(model.layers):
        if not layer.get_weights():  # Skip layers without weights
            continue

        # Zero out the weights of the current layer
        zeroed_weights = [np.zeros_like(w) for w in layer.get_weights()]
        layer.set_weights(zeroed_weights)

        # Evaluate accuracy with the current layer zeroed out
        _, new_accuracy = model.evaluate(*validation_data, verbose=0)
        accuracy_drop = original_accuracy - new_accuracy

        # Restore the original weights
        layer.set_weights(original_weights[i])

        # Record the accuracy impact for the layer
        layer_importance[layer.name] = accuracy_drop

    return layer_importance


def prune_layers_by_importance(model, layer_importance, pruning_fraction=0.2):
    """
    Prunes a fraction of layers based on their impact score.
    The less impactful layers (lower accuracy drop) are pruned first.
    """
    num_layers_to_prune = int(len(layer_importance) * pruning_fraction)
    sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1])

    for layer_name, _ in sorted_layers[:num_layers_to_prune]:
        for layer in model.layers:
            if layer.name == layer_name:
                # Zero out the least impactful layer's weights
                zeroed_weights = [np.zeros_like(w) for w in layer.get_weights()]
                layer.set_weights(zeroed_weights)
                print(f"Pruned layer: {layer.name}")

    return model

    
def fedavg(models):
    avg_weights = []
    for layer_index in range(len(models[0].layers)):
        layer_weights = [model.layers[layer_index].get_weights() for model in models]
        averaged_layer_weights = [np.mean(np.stack(weights), axis=0) for weights in zip(*layer_weights)]
        avg_weights.append(averaged_layer_weights)
    return avg_weights


def fedprox(models, global_model, mu=0.1):
    avg_weights = []
    for layer_index in range(len(global_model.layers)):
        print(f"fedprox layer index {layer_index}")
        # Get the weights for each model and the global model for the current layer
        layer_weights = [model.layers[layer_index].get_weights() for model in models]
        global_weights = global_model.layers[layer_index].get_weights()

        # Ensure prox_weights are computed component-wise for each part of the layer weights (e.g., kernel, bias)
        prox_weights = []
        for w_set, gw in zip(zip(*layer_weights), global_weights):
            prox_set = [(1 - mu) * w + mu * gw for w in w_set]
            prox_weights.append(np.mean(np.stack(prox_set), axis=0))

        avg_weights.append(prox_weights)
    return avg_weights


def weighted_fedavg(models, client_weights):
    avg_weights = []
    for layer_index in range(len(models[0].layers)):
        layer_weights = [model.layers[layer_index].get_weights() for model in models]
        weighted_layer = [np.average(np.stack(weights), axis=0, weights=client_weights) for weights in zip(*layer_weights)]
        avg_weights.append(weighted_layer)
    return avg_weights


# def evaluate_layer_importance(model, validation_data):
#     """
#     Zero out each layer individually and measure the drop in validation accuracy.
#     Returns a dictionary of layers with their corresponding impact scores.
#     """
#     layer_importance = {}
#     original_weights = [layer.get_weights() for layer in model.layers]
#     _, original_accuracy = model.evaluate(*validation_data, verbose=0)

#     for i, layer in enumerate(model.layers):
#         if not layer.get_weights():  # Skip layers without weights
#             continue

#         # Zero out the weights of the current layer
#         zeroed_weights = [np.zeros_like(w) for w in layer.get_weights()]
#         layer.set_weights(zeroed_weights)

#         # Evaluate accuracy with the current layer zeroed out
#         _, new_accuracy = model.evaluate(*validation_data, verbose=0)
#         accuracy_drop = original_accuracy - new_accuracy

#         # Restore the original weights
#         layer.set_weights(original_weights[i])

#         # Record the accuracy impact for the layer
#         layer_importance[layer.name] = accuracy_drop

#     return layer_importance


# def prune_layers_by_importance(model, layer_importance, pruning_fraction=0.2):
#     """
#     Prunes a fraction of layers based on their impact score.
#     The less impactful layers (lower accuracy drop) are pruned first.
#     """
#     num_layers_to_prune = int(len(layer_importance) * pruning_fraction)
#     sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1])

#     for layer_name, _ in sorted_layers[:num_layers_to_prune]:
#         for layer in model.layers:
#             if layer.name == layer_name:
#                 # Zero out the least impactful layer's weights
#                 zeroed_weights = [np.zeros_like(w) for w in layer.get_weights()]
#                 layer.set_weights(zeroed_weights)
#                 print(f"Pruned layer: {layer.name}")

#     return model


def set_model_weights(model_template, weights):
    for layer_index, layer in enumerate(model_template.layers):
        layer_weights = layer.get_weights()
        
        if len(layer_weights) != len(weights[layer_index]):
            print(f"Skipping layer '{layer.name}' due to weight mismatch.")
            continue

        try:
            layer.set_weights(weights[layer_index])
        except ValueError as e:
            print(f"Could not set weights for layer '{layer.name}': {e}")



def load_validation_data():
    (_, _), validation_data, (_, _) = load_fashionmnist_data(validation_split=0.2)
    return validation_data


def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Model Pruning")
    parser.add_argument(
        "--aggregation", choices=["fedavg", "fedprox", "weighted_fedavg", "pruned_fedavg"], required=True,
        help="Select aggregation method for federated learning"
    )
    parser.add_argument("--pruning_fraction", type=float, default=0.2,
                        help="Fraction of layers to prune based on accuracy impact (for pruned_fedavg only)")

    args = parser.parse_args()
    
    model_dir = "models/"
    clients_file = "successful_clients.json"
    output_model_path = "models/final_model.keras"
    quantization_factor = 256

    # Load successful clients and their models
    with open(clients_file, "r") as f:
        successful_clients = json.load(f)["successful_clients"]
    models = load_models_from_selected_clients(model_dir, successful_clients)
    validation_data = load_validation_data()

    # Select the aggregation method
    if args.aggregation == "fedavg":
        avg_weights = fedavg(models)
    elif args.aggregation == "fedprox":
        global_model = models[0]  # Assume a global model for FedProx
        avg_weights = fedprox(models, global_model)
    elif args.aggregation == "weighted_fedavg":
        # Assume some client-specific weights; here, we use equal weights as a placeholder
        client_weights = [1.0] * len(models)
        avg_weights = weighted_fedavg(models, client_weights)
    elif args.aggregation == "pruned_fedavg":
        # Evaluate importance of each layer in the global model
        global_model = models[0]
        layer_importance = evaluate_layer_importance(global_model, validation_data)
        print("Layer importance scores:", layer_importance)
        # Prune the least impactful layers
        pruned_model = prune_layers_by_importance(global_model, layer_importance, args.pruning_fraction)
        avg_weights = pruned_model.get_weights()
    
    # Save the final pruned/aggregated model
    final_model = models[0]
    set_model_weights(final_model, avg_weights)
    final_model.save(output_model_path)
    print(f"Final model saved at: {output_model_path}")

    # Evaluate the final model
    val_loss, val_accuracy = final_model.evaluate(*validation_data, verbose=2)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")


if __name__ == "__main__":
    main()
