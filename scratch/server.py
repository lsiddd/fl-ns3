import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load FashionMNIST dataset


def load_fashionmnist_data(validation_split=0.2):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Reshape to add channel dimension (FashionMNIST is grayscale, so 1 channel)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Split the training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=validation_split, random_state=42
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# Function to load models from a directory


def load_models_from_directory(directory):
    models = []
    model_filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".keras"):
            model_path = os.path.join(directory, filename)
            model = tf.keras.models.load_model(model_path)
            models.append(model)
            model_filenames.append(model_path)
    return models, model_filenames


# FedAvg: Averages the weights of the models


def fedavg(models):
    # Get the weights from each model
    weights = [model.get_weights() for model in models]

    # Initialize an empty list to store averaged weights
    avg_weights = []

    # Iterate through each layer's weights
    for weights_per_layer in zip(*weights):
        # Convert weights_per_layer into numpy array for averaging
        layer_stack = np.stack(weights_per_layer, axis=0)

        # Average the weights for each layer
        avg_layer_weights = np.mean(layer_stack, axis=0)

        # Append the averaged weights to the list
        avg_weights.append(avg_layer_weights)

    return avg_weights


# Function to quantize weights


def quantize_weights(weights, quantization_levels):
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


# Function to compress the quantized weights using zlib


def compress_weights(weights):
    # Flatten the weights to compress
    flat_weights = np.concatenate([w.flatten() for w in weights])

    # Convert to bytes
    weight_bytes = flat_weights.tobytes()

    # Compress using zlib
    compressed_weights = zlib.compress(weight_bytes)

    return compressed_weights


# Function to save the FedAvg model and replace individual models


def save_and_replace_fedavg_model(
    directory, output_path, validation_data, quantization_levels=256
):
    # Load all models from the directory
    models, model_filenames = load_models_from_directory(directory)

    if len(models) == 0:
        raise ValueError("No models found in the directory with .keras extension.")

    # Perform FedAvg to average the weights
    avg_weights = fedavg(models)

    # Quantize the weights
    quantized_weights = quantize_weights(avg_weights, quantization_levels)

    # Compress the quantized weights
    compressed_weights = compress_weights(quantized_weights)

    # Get the size of the compressed weights
    compressed_size = len(compressed_weights)
    print(f"Compressed model size: {compressed_size} bytes")

    # Create a new model with the same architecture as one of the models
    model_template = models[0]

    # Set the quantized weights to the new model
    model_template.set_weights(quantized_weights)

    # Save the new FedAvg model
    model_template.save(output_path)
    print(f"FedAvg model saved at: {output_path}")

    # Evaluate the model on the validation set
    x_val, y_val = validation_data
    val_loss, val_accuracy = model_template.evaluate(x_val, y_val, verbose=2)

    # Replace individual client models with the aggregated model
    for model_path in model_filenames:
        # Save the aggregated weights in place of the original model
        model_template.save(model_path)
        print(f"Replaced client model at: {model_path}")

    with open("metrics.txt", "a+") as metrics_file:
        metrics_file.write(f"Validation Loss: {val_loss}\n")
        metrics_file.write(f"Validation Accuracy: {val_accuracy}\n")

    return val_loss, val_accuracy, compressed_size


# Main workflow
if __name__ == "__main__":
    # Paths
    directory_path = "models/"  # Replace with your directory containing .h5 models
    # Replace with desired output path
    output_model_path = "models/fedavg_model.keras"

    # Load FashionMNIST data and get validation split
    (_, _), (x_val, y_val), (_, _) = load_fashionmnist_data(validation_split=0.2)

    # Save and replace individual models with the FedAvg model
    val_loss, val_accuracy, compressed_size = save_and_replace_fedavg_model(
        directory_path, output_model_path, (x_val, y_val), quantization_levels=256
    )

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Compressed Model Size: {compressed_size} bytes")
