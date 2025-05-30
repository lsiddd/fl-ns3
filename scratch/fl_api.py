# Filename: scratch/sim/fl_api.py
import collections
import time
import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, request, jsonify
import threading
import logging # Import logging
import socket # Import socket for port checking

# --- Configuração Inicial e Supressão de Logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Configure Flask logging
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO) # Default level for application logs

# You can set a more verbose level for specific debug scenarios:
# app.logger.setLevel(logging.DEBUG)

# --- Estado Global da Simulação FL ---
FL_STATE = {
    "config": None,
    "client_train_datasets": None,
    "client_num_samples_unique": None,
    "centralized_test_dataset": None,
    "num_classes": None,
    "global_model": None,
    "current_global_weights": None,
    "eligible_client_indices": None,
    "history_log": collections.defaultdict(list),
    "current_round": 0,
    "simulation_initialized": False,
    "model_compiled": False,
    "data_loaded": False,
    "is_training_round_active": False # Para evitar chamadas concorrentes de run_round
}

# --- 0. Funções de Argumentos (adaptadas para defaults) ---
def get_default_args():
    parser = argparse.ArgumentParser(description="Simulador de Aprendizado Federado Manual Avançado")
    # Parâmetros do Dataset e Particionamento
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'emnist_digits', 'emnist_char', 'cifar10'])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--non_iid_type', type=str, default='iid', choices=['iid', 'pathological', 'dirichlet'])
    parser.add_argument('--non_iid_alpha', type=float, default=0.5) # Para Pathological: num_classes_per_client
    parser.add_argument('--quantity_skew_type', type=str, default='uniform', choices=['uniform', 'power_law'])
    parser.add_argument('--power_law_beta', type=float, default=2.0)
    parser.add_argument('--feature_skew_type', type=str, default='none', choices=['none', 'noise'])
    parser.add_argument('--noise_std_dev', type=float, default=0.1)

    # Parâmetros de Treinamento Federado
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--num_rounds_api_max', type=int, default=100, help='Max rounds for API, actual rounds controlled by /run_round calls') # Renomeado de num_rounds
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--client_lr', type=float, default=0.01)
    parser.add_argument('--aggregation_method', type=str, default='fedavg', choices=['fedavg'])

    # Outros
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    # Networking parameters (add port here)
    parser.add_argument('--port', type=int, default=5000, help='Starting port for the Flask API server')

    default_args = parser.parse_args([])
    app.logger.debug(f"Default arguments initialized: {vars(default_args)}")
    return default_args

# --- 1. Definição do Modelo Keras ---
def create_model_api(dataset_name, num_classes_override=None, seed=None, config=None): # Adicionado config
    if config is None: config = FL_STATE.get('config', get_default_args()) # Fallback

    if dataset_name == 'mnist' or dataset_name == 'emnist_digits':
        input_shape = (28, 28, 1)
        num_classes = 10
    elif dataset_name == 'emnist_char':
        input_shape = (28, 28, 1)
        num_classes = 62
    elif dataset_name == 'cifar10':
        input_shape = (32, 32, 3)
        num_classes = 10
    else:
        app.logger.error(f"Dataset desconhecido para criação de modelo: {dataset_name}")
        raise ValueError(f"Dataset desconhecido para criação de modelo: {dataset_name}")

    if num_classes_override is not None:
        num_classes = num_classes_override

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape,
                                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if dataset_name == 'cifar10':
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)))
    app.logger.info(f"Model created for dataset '{dataset_name}' with input shape {input_shape} and {num_classes} classes.")
    return model

# --- 2. Carregamento e Pré-processamento de Dados (adaptado para usar FL_STATE['config']) ---
def preprocess_dataset_for_model_creation_api(dataset_element):
    image = tf.cast(dataset_element['image'], tf.float32) / 255.0
    label = dataset_element['label']
    if image.shape[-1] != 1 and image.shape[-1] != 3:
        image = tf.expand_dims(image, axis=-1)
    return (image, label)

def create_client_tf_dataset_api(client_x_data, client_y_data, config):
    def format_for_fit(element):
        image = tf.cast(element['image'], tf.float32) / 255.0
        if config.dataset in ['mnist', 'emnist_digits', 'emnist_char'] and len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        return (image, element['label'])

    client_tf_ds = tf.data.Dataset.from_tensor_slices({'image': client_x_data, 'label': client_y_data})
    client_tf_ds = client_tf_ds.map(format_for_fit, num_parallel_calls=tf.data.AUTOTUNE)

    num_unique_samples_for_client = len(client_y_data) # Derivado de len(client_y_data)
    if num_unique_samples_for_client > 0:
        client_tf_ds = client_tf_ds.shuffle(buffer_size=num_unique_samples_for_client, seed=config.seed, reshuffle_each_iteration=True)
        client_tf_ds = client_tf_ds.repeat(config.local_epochs)
    app.logger.debug(f"Client dataset created with {num_unique_samples_for_client} unique samples, repeated {config.local_epochs} times, batch size {config.batch_size}.")
    return client_tf_ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)


def load_and_distribute_data_api():
    config = FL_STATE['config']
    if config.seed is not None:
        np.random.seed(config.seed)
        # tf.keras.utils.set_random_seed(config.seed) # Chamado em /initialize_simulation

    if config.dataset == 'mnist': tfds_name = 'mnist'
    elif config.dataset == 'emnist_digits': tfds_name = 'emnist/digits'
    elif config.dataset == 'emnist_char': tfds_name = 'emnist/byclass'
    elif config.dataset == 'cifar10': tfds_name = 'cifar10'
    else:
        app.logger.error(f"Dataset {config.dataset} não suportado.")
        raise ValueError(f"Dataset {config.dataset} não suportado.")

    app.logger.info(f"Loading dataset '{tfds_name}'...")
    train_ds_global_tfds, ds_info = tfds.load(tfds_name, split='train', as_supervised=False, with_info=True, shuffle_files=True)
    test_ds_global_tfds = tfds.load(tfds_name, split='test', as_supervised=False)
    num_classes = ds_info.features['label'].num_classes
    total_train_samples_tfds = ds_info.splits['train'].num_examples
    app.logger.info(f"Dataset '{tfds_name}' loaded. Total train samples: {total_train_samples_tfds}, Num classes: {num_classes}.")

    app.logger.info(f"Converting {total_train_samples_tfds} train samples from TFDS to NumPy...")
    train_samples_np = list(tfds.as_numpy(train_ds_global_tfds))
    x_global_orig = np.array([s['image'] for s in train_samples_np])
    y_global_orig = np.array([s['label'] for s in train_samples_np])
    if y_global_orig.ndim > 1 and y_global_orig.shape[1] == 1:
        y_global_orig = y_global_orig.flatten()
    app.logger.info("Conversion to NumPy complete.")

    client_data_indices_map = [[] for _ in range(config.num_clients)]

    # A. Skew de Labels
    if config.non_iid_type == 'iid':
        app.logger.info("Applying IID data partitioning.")
        all_indices = np.arange(total_train_samples_tfds)
        np.random.shuffle(all_indices)
        idx_splits = np.array_split(all_indices, config.num_clients)
        for i in range(config.num_clients): client_data_indices_map[i] = idx_splits[i].tolist()

    elif config.non_iid_type == 'pathological':
        num_classes_per_client = int(max(1, config.non_iid_alpha)) # alpha is num_classes_per_client
        app.logger.info(f"Applying Pathological non-IID partitioning: ~{num_classes_per_client} classes per client.")
        indices_by_label = [np.where(y_global_orig == i)[0] for i in range(num_classes)]
        for idx_list in indices_by_label: np.random.shuffle(idx_list)

        shards_per_label = []
        # The number of shards per label should ensure all data is eventually used
        # A common approach is to create N shards per class, where N is sufficient
        # to distribute among clients. E.g., if each client gets 2 classes, and there are 10 clients,
        # and 10 classes, we need 20 shards total.
        # A simpler way for pathological is to create 2 shards per class, or just assign 2 classes directly

        # Create shards for each class
        all_shards = []
        for label_idx, indices_list in enumerate(indices_by_label):
            # Create a few shards per label to distribute
            num_shards_for_this_label = max(1, config.num_clients // num_classes) # Try to have enough for all clients
            if num_shards_for_this_label < 2: num_shards_for_this_label = 2 # At least 2 shards per class

            if len(indices_list) > 0:
                shards = np.array_split(indices_list, num_shards_for_this_label)
                for shard in shards:
                    if len(shard) > 0: all_shards.append((label_idx, shard.tolist()))

        np.random.shuffle(all_shards) # Shuffle all shards

        # Assign shards to clients
        client_data_indices_map = [[] for _ in range(config.num_clients)]
        client_assigned_labels = [set() for _ in range(config.num_clients)]

        shard_idx = 0
        while shard_idx < len(all_shards):
            # First pass: assign distinct labels up to num_classes_per_client
            for client_id in range(config.num_clients):
                if shard_idx >= len(all_shards): break
                current_shard_label, current_shard_indices = all_shards[shard_idx]

                if len(client_assigned_labels[client_id]) < num_classes_per_client or current_shard_label in client_assigned_labels[client_id]:
                    # If client hasn't reached its class limit, or if this is a label it already has (allowing more data of same class)
                    client_data_indices_map[client_id].extend(current_shard_indices)
                    client_assigned_labels[client_id].add(current_shard_label)
                    shard_idx += 1
                # If client reached class limit AND this shard is a new label, skip for now to prioritize others
                # This makes it hard to use all data. Simple round robin is better.

            # If still shards left after trying to distribute distinct labels, just round-robin remaining
            # This logic is complex to ensure all data is used and IID/non-IID properties are strict.
            # For simplicity, if a client gets `num_classes_per_client` shards, just add more data to them.
            if shard_idx < len(all_shards): # If some shards are left unassigned from the first pass
                for client_id in range(config.num_clients):
                    if shard_idx >= len(all_shards): break
                    current_shard_label, current_shard_indices = all_shards[shard_idx]
                    client_data_indices_map[client_id].extend(current_shard_indices)
                    client_assigned_labels[client_id].add(current_shard_label)
                    shard_idx += 1

        for client_id in range(config.num_clients):
            client_data_indices_map[client_id] = list(np.unique(np.array(client_data_indices_map[client_id])))
            app.logger.debug(f"  Client {client_id}: assigned {len(client_data_indices_map[client_id])} samples, labels: {sorted(list(client_assigned_labels[client_id]))}")

    elif config.non_iid_type == 'dirichlet':
        app.logger.info(f"Applying Dirichlet non-IID partitioning with alpha={config.non_iid_alpha}.")
        label_proportions = np.random.dirichlet([config.non_iid_alpha] * num_classes, config.num_clients)
        indices_by_label = [np.where(y_global_orig == i)[0] for i in range(num_classes)]
        for idx_list in indices_by_label: np.random.shuffle(idx_list)

        ptr_by_label = [0] * num_classes

        # Calculate target samples per client for each class
        client_target_samples_per_class = np.zeros((config.num_clients, num_classes), dtype=int)

        # Distribute total samples based on Dirichlet proportions
        # This approach ensures all samples are used if total_train_samples_tfds is the base.
        total_samples_assigned_to_clients_so_far = 0
        for client_idx in range(config.num_clients):
            # total_samples_for_this_client = int(label_proportions[client_idx].sum() * total_train_samples_tfds / config.num_clients)
            # This is tricky: sum of proportions for a client is 1. We need to normalize proportions by column.
            # A common way for Dirichlet:
            # 1. Calculate how many samples of *each class* go to *each client*
            # 2. Sum for each client.

            # This loop iterates through clients, and for each client, calculates how many samples it gets from *each* class.
            # This is slightly different from some other Dirichlet implementations where you iterate class by class.
            # The original code's logic of `target_num_samples_from_class_for_client` implies this.

            # Total samples available for distribution for each label
            available_samples_per_label = [len(indices) for indices in indices_by_label]

            client_indices_list = []
            for label_k in range(num_classes):
                if available_samples_per_label[label_k] == 0: continue

                # Normalize the client's proportion for this class across all clients.
                # Sum of label_proportions[client_idx, label_k] for a fixed client_idx over all label_k is 1.0.
                # Sum of label_proportions[client_idx, label_k] for a fixed label_k over all client_idx is not 1.0.
                # We need to scale label_proportions[client_idx, label_k] by the *total amount of data* in that class,
                # then ensure the sum over clients for that class doesn't exceed total_samples_in_that_class.

                # Let's try the simple and common way:
                # Assign fraction of total samples based on client's Dirichlet vector
                num_samples_for_this_client_overall = total_train_samples_tfds / config.num_clients # Base for each client

                # Take samples for this client from each class based on its proportion
                num_to_take_from_label_k = int(label_proportions[client_idx, label_k] * num_samples_for_this_client_overall)

                start_ptr = ptr_by_label[label_k]
                num_available_for_label_k = len(indices_by_label[label_k]) - start_ptr

                num_actually_taken = min(num_to_take_from_label_k, num_available_for_label_k)

                if num_actually_taken > 0:
                    end_ptr = start_ptr + num_actually_taken
                    client_indices_list.extend(indices_by_label[label_k][start_ptr:end_ptr])
                    ptr_by_label[label_k] = end_ptr

            np.random.shuffle(client_indices_list)
            client_data_indices_map[client_idx] = client_indices_list
            total_samples_assigned_to_clients_so_far += len(client_indices_list)
            app.logger.debug(f"  Client {client_idx}: assigned {len(client_data_indices_map[client_idx])} samples.")

        app.logger.info(f"Total samples distributed via Dirichlet: {total_samples_assigned_to_clients_so_far} out of {total_train_samples_tfds}.")

    else:
        app.logger.error(f"Tipo de não-IID desconhecido: {config.non_iid_type}")
        raise ValueError(f"Tipo de não-IID desconhecido: {config.non_iid_type}")

    # B. Aplicar Skew de Quantidade
    if config.quantity_skew_type == 'power_law':
        app.logger.info(f"Applying Power Law Quantity Skew with beta={config.power_law_beta}.")
        min_samples_per_client = max(1, config.batch_size // 4)
        raw_power_law_samples = np.random.pareto(config.power_law_beta, config.num_clients) + 1e-6
        proportions = raw_power_law_samples / np.sum(raw_power_law_samples)

        total_samples_currently_assigned = sum(len(idx_list) for idx_list in client_data_indices_map)
        if total_samples_currently_assigned == 0:
            app.logger.warning("No samples currently assigned, cannot apply power law quantity skew.")
            # Skip quantity skew if no data
        else:
            target_samples_per_client = (proportions * total_samples_currently_assigned).astype(int)
            target_samples_per_client = np.maximum(target_samples_per_client, min_samples_per_client)

            # Adjust to ensure sum does not exceed total available (important if previous steps reduced total)
            current_sum_target = np.sum(target_samples_per_client)
            if current_sum_target > total_samples_currently_assigned:
                excess_ratio = total_samples_currently_assigned / current_sum_target
                target_samples_per_client = (target_samples_per_client * excess_ratio).astype(int)
                target_samples_per_client = np.maximum(target_samples_per_client, min_samples_per_client) # Re-apply min

            new_client_data_indices_map = []
            for c_idx in range(config.num_clients):
                current_indices_for_client = np.array(client_data_indices_map[c_idx])
                np.random.shuffle(current_indices_for_client)
                target_count = target_samples_per_client[c_idx]

                if len(current_indices_for_client) > target_count:
                    new_client_data_indices_map.append(current_indices_for_client[:target_count].tolist())
                else:
                    new_client_data_indices_map.append(current_indices_for_client.tolist())
            client_data_indices_map = new_client_data_indices_map
            app.logger.info(f"  Power Law samples per client after adjustment: {target_samples_per_client.tolist()}")

    # C. Criar tf.data.Datasets
    client_datasets_train_list = []
    client_num_samples_unique_list = []

    app.logger.info("Creating TensorFlow datasets for each client...")
    for c_idx in range(config.num_clients):
        indices_for_this_client = np.array(client_data_indices_map[c_idx], dtype=int)
        num_unique_samples = len(indices_for_this_client)
        client_num_samples_unique_list.append(num_unique_samples)

        if num_unique_samples == 0:
            empty_x = np.array([], dtype=x_global_orig.dtype).reshape(0, *x_global_orig.shape[1:])
            empty_y = np.array([], dtype=y_global_orig.dtype)
            client_tf_ds = create_client_tf_dataset_api(empty_x, empty_y, config)
            client_datasets_train_list.append(client_tf_ds)
            app.logger.debug(f"  Client {c_idx}: 0 unique samples, created empty dataset.")
            continue

        client_x_data_orig = x_global_orig[indices_for_this_client]
        client_y_data_orig = y_global_orig[indices_for_this_client]

        client_x_processed = client_x_data_orig.astype('float32')
        if config.feature_skew_type == 'noise' and config.noise_std_dev > 0:
            noise_abs_std_dev = config.noise_std_dev * 255.0
            noise = np.random.normal(0, noise_abs_std_dev, client_x_processed.shape).astype(client_x_processed.dtype)
            client_x_processed = client_x_processed + noise
            client_x_processed = np.clip(client_x_processed, 0.0, 255.0)
            app.logger.debug(f"  Client {c_idx}: applied feature skew (noise) with std_dev {config.noise_std_dev}.")

        client_tf_ds = create_client_tf_dataset_api(client_x_processed, client_y_data_orig, config)
        client_datasets_train_list.append(client_tf_ds)
        app.logger.debug(f"  Client {c_idx}: assigned {num_unique_samples} samples.")

    centralized_test_data = test_ds_global_tfds.map(preprocess_dataset_for_model_creation_api, num_parallel_calls=tf.data.AUTOTUNE)\
                                               .batch(config.batch_size * 2)\
                                               .prefetch(tf.data.AUTOTUNE)

    app.logger.info(f"Data distribution completed. Samples per client: {client_num_samples_unique_list}")

    FL_STATE['client_train_datasets'] = client_datasets_train_list
    FL_STATE['client_num_samples_unique'] = client_num_samples_unique_list
    FL_STATE['centralized_test_dataset'] = centralized_test_data
    FL_STATE['num_classes'] = num_classes
    FL_STATE['eligible_client_indices'] = [i for i, n_samples in enumerate(client_num_samples_unique_list) if n_samples > 0]
    FL_STATE['data_loaded'] = True
    app.logger.info(f"Eligible clients (with >0 samples): {len(FL_STATE['eligible_client_indices'])} out of {config.num_clients}.")

    return True # Success

# --- 3. Treinamento e Agregação Federada (adaptado para usar FL_STATE['config']) ---
def client_update_api(model_template, global_weights, client_tf_dataset, num_unique_samples_this_client):
    config = FL_STATE['config']
    if num_unique_samples_this_client == 0:
        app.logger.warning("  Client update skipped: 0 unique samples for this client.")
        return global_weights, 0.0, 0.0

    client_model = tf.keras.models.clone_model(model_template) # Clona a estrutura

    if config.client_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.client_lr)
    elif config.client_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.client_lr)
    else:
        app.logger.error(f"Otimizador de cliente desconhecido: {config.client_optimizer}")
        raise ValueError(f"Otimizador de cliente desconhecido: {config.client_optimizer}")

    client_model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    client_model.set_weights(global_weights) # Agora define os pesos

    steps_per_epoch_val = None
    if num_unique_samples_this_client > 0:
         steps_per_epoch_val = int(np.ceil(num_unique_samples_this_client * config.local_epochs / config.batch_size))

    app.logger.debug(f"  Client training for {config.local_epochs} local epochs, steps_per_epoch: {steps_per_epoch_val}.")
    history = client_model.fit(client_tf_dataset, epochs=1, verbose=0, steps_per_epoch=steps_per_epoch_val)

    loss = history.history['loss'][-1] if 'loss' in history.history and history.history['loss'] else 0.0
    accuracy = history.history['sparse_categorical_accuracy'][-1] if 'sparse_categorical_accuracy' in history.history and history.history['sparse_categorical_accuracy'] else 0.0
    app.logger.debug(f"  Client training finished. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}.")
    return client_model.get_weights(), loss, accuracy

def aggregate_weights_fedavg_api(client_weights_list, client_num_samples_list):
    if not client_weights_list:
        app.logger.warning("  Aggregation skipped: No client weights to aggregate.")
        return None
    total_samples_in_round = sum(client_num_samples_list)
    if total_samples_in_round == 0:
        app.logger.warning("  Aggregation skipped: No data samples from participating clients in this round.")
        return client_weights_list[0] if client_weights_list else None # Return first client's weights or None

    app.logger.info(f"  Aggregating weights using FedAvg. Total samples: {total_samples_in_round}.")
    avg_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    for i, client_weights in enumerate(client_weights_list):
        num_samples = client_num_samples_list[i]
        if num_samples == 0: continue
        weight_factor = num_samples / total_samples_in_round
        for layer_idx, layer_weights in enumerate(client_weights):
            avg_weights[layer_idx] += weight_factor * layer_weights
    app.logger.info("  FedAvg aggregation complete.")
    return avg_weights

# --- Endpoints da API Flask ---

@app.route('/configure', methods=['POST'])
def configure_simulation():
    global FL_STATE
    if FL_STATE['simulation_initialized'] or FL_STATE['is_training_round_active']:
        app.logger.warning("Attempted /configure while simulation active. Returning 400.")
        return jsonify({"error": "A simulação já está em andando ou inicializada. Use /reset_simulation primeiro."}), 400

    try:
        data = request.get_json()
        if not data:
            app.logger.error("No JSON payload provided to /configure.")
            return jsonify({"error": "Payload JSON não fornecido ou inválido."}), 400

        app.logger.info(f"Received configuration request: {data}")
        # Começa com defaults e sobrescreve com o que foi passado
        current_args = get_default_args()
        # Valida e atualiza os argumentos
        for key, value in data.items():
            if hasattr(current_args, key):
                default_val = getattr(current_args, key)
                if default_val is not None:
                    try:
                        # Special handling for 'port' if needed later, but for now it's just a value.
                        # The port used for running is handled in __main__
                        setattr(current_args, key, type(default_val)(value))
                    except (ValueError, TypeError) as e:
                         app.logger.error(f"Invalid value for '{key}': {value}. Expected type {type(default_val)}. Error: {e}")
                         return jsonify({"error": f"Valor inválido para '{key}': {value}. Esperado tipo {type(default_val)}. Erro: {e}"}), 400
                else: # Se o default é None, aceita o tipo que veio
                    setattr(current_args, key, value)
            else:
                app.logger.warning(f"Unknown configuration key '{key}' ignored.")

        FL_STATE['config'] = current_args
        FL_STATE['simulation_initialized'] = False # Resetar flags dependentes
        FL_STATE['model_compiled'] = False
        FL_STATE['data_loaded'] = False
        FL_STATE['current_round'] = 0
        FL_STATE['history_log'] = collections.defaultdict(list)

        app.logger.info(f"Simulation configured successfully. Current config: {vars(FL_STATE['config'])}")
        return jsonify({"message": "Configuração recebida com sucesso.", "config": vars(FL_STATE['config'])}), 200
    except Exception as e:
        app.logger.exception(f"Error in /configure: {e}") # Logs full traceback
        return jsonify({"error": f"Erro interno ao processar configuração: {str(e)}"}), 500

@app.route('/initialize_simulation', methods=['POST'])
def initialize_simulation():
    global FL_STATE
    if FL_STATE['is_training_round_active']:
        app.logger.warning("Attempted /initialize_simulation while training round is active.")
        return jsonify({"error": "Uma rodada de treinamento está ativa."}), 400
    if FL_STATE['simulation_initialized']:
        app.logger.info("Simulation already initialized. Returning 200.")
        return jsonify({"message": "Simulação já inicializada. Use /reset_simulation para reconfigurar."}), 200
    if FL_STATE['config'] is None:
        app.logger.error("Configuration not defined before /initialize_simulation.")
        return jsonify({"error": "Configuração não definida. Chame /configure primeiro."}), 400

    try:
        app.logger.info("Initializing simulation...")
        # Definir sementes globais
        if FL_STATE['config'].seed is not None:
            tf.keras.utils.set_random_seed(FL_STATE['config'].seed)
            np.random.seed(FL_STATE['config'].seed)
            app.logger.info(f"Global seed set to {FL_STATE['config'].seed}.")

        # 1. Carregar e distribuir dados
        app.logger.info("  Loading and distributing data...")
        data_load_success = load_and_distribute_data_api()
        if not data_load_success:
            app.logger.error("  Failed to load/distribute data during initialization.")
            return jsonify({"error": "Falha ao carregar/distribuir dados."}), 500
        app.logger.info(f"  Data loaded. Eligible clients: {FL_STATE['eligible_client_indices']}")

        # 2. Criar e compilar modelo global
        app.logger.info("  Creating global model...")
        FL_STATE['global_model'] = create_model_api(
            FL_STATE['config'].dataset,
            num_classes_override=FL_STATE['num_classes'],
            seed=FL_STATE['config'].seed,
            config=FL_STATE['config']
        )
        FL_STATE['global_model'].compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        FL_STATE['current_global_weights'] = FL_STATE['global_model'].get_weights()
        FL_STATE['model_compiled'] = True
        app.logger.info("  Global model created and compiled.")

        FL_STATE['simulation_initialized'] = True
        FL_STATE['current_round'] = 0
        FL_STATE['history_log'] = collections.defaultdict(list) # Limpa histórico

        initial_loss, initial_acc = float('nan'), float('nan')
        if FL_STATE['centralized_test_dataset'] and FL_STATE['global_model']:
            app.logger.info("  Evaluating initial global model...")
            initial_loss, initial_acc = FL_STATE['global_model'].evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
            FL_STATE['history_log']['round'].append(0)
            FL_STATE['history_log']['avg_client_loss'].append(float('nan')) # No client training in round 0
            FL_STATE['history_log']['avg_client_acc'].append(float('nan'))
            FL_STATE['history_log']['global_test_loss'].append(initial_loss)
            FL_STATE['history_log']['global_test_acc'].append(initial_acc)
            app.logger.info(f"  Initial Model (Round 0) - Test Loss: {initial_loss:.4f}, Test Accuracy: {initial_acc:.4f}")
        else:
            app.logger.warning("  Cannot evaluate initial model: centralized test dataset or global model not available.")

        return jsonify({
            "message": "Simulação FL inicializada com sucesso.",
            "num_clients_total": FL_STATE['config'].num_clients,
            "num_eligible_clients": len(FL_STATE['eligible_client_indices']),
            "client_samples_distribution": FL_STATE['client_num_samples_unique'],
            "initial_model_evaluation": {
                "loss": initial_loss,
                "accuracy": initial_acc
            }
        }), 200

    except Exception as e:
        app.logger.exception(f"Error in /initialize_simulation: {e}")
        FL_STATE['simulation_initialized'] = False # Reverte em caso de erro
        return jsonify({"error": f"Erro interno ao inicializar simulação: {str(e)}"}), 500

@app.route('/run_round', methods=['POST'])
def run_training_round():
    global FL_STATE
    if FL_STATE['is_training_round_active']:
        app.logger.warning("Attempted /run_round while another round is active.")
        return jsonify({"error": "Outra rodada de treinamento já está em andamento."}), 409 # Conflict
    if not FL_STATE['simulation_initialized']:
        app.logger.error("Attempted /run_round before simulation initialized.")
        return jsonify({"error": "Simulação não inicializada. Chame /initialize_simulation primeiro."}), 400
    if not FL_STATE['eligible_client_indices']:
        app.logger.warning("No eligible clients available for training in /run_round.")
        return jsonify({"message": "Nenhum cliente elegível com dados para treinamento. Rodada pulada."}), 200

    FL_STATE['is_training_round_active'] = True
    try:
        config = FL_STATE['config']
        FL_STATE['current_round'] += 1
        round_num = FL_STATE['current_round']

        app.logger.info(f"\n--- Iniciando Rodada de Treinamento FL: {round_num} ---")
        round_start_time = time.time()

        data = request.get_json(silent=True) or {}
        selected_client_indices_input = data.get('client_indices', None)

        sampled_original_client_indices = []
        if selected_client_indices_input is not None:
            if not isinstance(selected_client_indices_input, list) or not all(isinstance(i, int) for i in selected_client_indices_input):
                FL_STATE['is_training_round_active'] = False
                app.logger.error(f"Invalid client_indices format: {selected_client_indices_input}")
                return jsonify({"error": "client_indices deve ser uma lista de inteiros."}), 400

            # Valida se os índices fornecidos são elegíveis
            sampled_original_client_indices = [
                idx for idx in selected_client_indices_input if idx in FL_STATE['eligible_client_indices']
            ]
            if len(sampled_original_client_indices) != len(selected_client_indices_input):
                app.logger.warning(f"Some manually selected clients ({len(selected_client_indices_input) - len(sampled_original_client_indices)}) are not eligible or do not exist. Using only valid ones: {sampled_original_client_indices}.")
            if not sampled_original_client_indices:
                FL_STATE['is_training_round_active'] = False
                app.logger.warning("No valid clients from manually selected indices for training.")
                return jsonify({"message": "Nenhum dos clientes selecionados manualmente é elegível ou possui dados. Rodada pulada."}), 200
            app.logger.info(f"  Clients selected by ns-3 (original indices): {sampled_original_client_indices}")
        else:
            num_clients_to_sample = min(config.clients_per_round, len(FL_STATE['eligible_client_indices']))
            if num_clients_to_sample > 0:
                sampled_original_client_indices = np.random.choice(
                    FL_STATE['eligible_client_indices'],
                    size=num_clients_to_sample,
                    replace=False
                ).tolist()
                app.logger.info(f"  Clients sampled by API (original indices): {sampled_original_client_indices}")
            else:
                 FL_STATE['is_training_round_active'] = False
                 app.logger.info("  No clients to sample for this round.")
                 return jsonify({"message": "Nenhum cliente para amostrar nesta rodada (verifique clients_per_round e dados dos clientes)."}), 200


        app.logger.info(f"  Round {round_num}/{getattr(config, 'num_rounds_api_max', 'N/A')} - Sampled clients (original indices): {sampled_original_client_indices}")

        current_round_client_updates = []
        current_round_client_sample_counts = []
        current_round_client_losses = []
        current_round_client_accuracies = []
        current_round_client_performance = {} # Detailed per-client performance

        for client_original_idx in sampled_original_client_indices:
            client_ds_for_training = FL_STATE['client_train_datasets'][client_original_idx]
            num_unique_samples = FL_STATE['client_num_samples_unique'][client_original_idx]

            if num_unique_samples == 0:
                app.logger.warning(f"    Client {client_original_idx} has 0 samples. Skipping training.")
                continue

            app.logger.info(f"    Training client {client_original_idx} with {num_unique_samples} samples...")
            updated_w, loss, acc = client_update_api(
                FL_STATE['global_model'],
                FL_STATE['current_global_weights'],
                client_ds_for_training,
                num_unique_samples
            )
            current_round_client_updates.append(updated_w)
            current_round_client_sample_counts.append(num_unique_samples)
            current_round_client_losses.append(loss)
            current_round_client_accuracies.append(acc)
            current_round_client_performance[client_original_idx] = {"loss": float(loss), "accuracy": float(acc), "num_samples": num_unique_samples}

            # --- Simulate Training Time and Model Size ---
            # Model size is usually constant for a given architecture
            simulated_model_size_bytes = 2 * 1024 * 1024 # 2MB example constant size
            # Training time scales with number of samples and local epochs
            simulated_training_time_ms = max(100, int(100 + num_unique_samples * 0.5 * config.local_epochs)) # Example: 100ms base + 0.5ms/sample/epoch
            current_round_client_performance[client_original_idx]['simulated_model_size_bytes'] = simulated_model_size_bytes
            current_round_client_performance[client_original_idx]['simulated_training_time_ms'] = simulated_training_time_ms
            app.logger.info(f"      Client {client_original_idx} - Loss: {loss:.4f}, Accuracy: {acc:.4f}, Samples: {num_unique_samples}, Estimated Time: {simulated_training_time_ms}ms, Estimated Size: {simulated_model_size_bytes} bytes")
            # --- End Simulation ---


        avg_loss_this_round = float('nan')
        avg_acc_this_round = float('nan')

        if not current_round_client_updates:
            app.logger.warning(f"    No clients trained in this round. Global weights remain unchanged.")
        elif config.aggregation_method == 'fedavg':
            new_global_weights = aggregate_weights_fedavg_api(current_round_client_updates, current_round_client_sample_counts)
            if new_global_weights is not None:
                FL_STATE['current_global_weights'] = new_global_weights
                FL_STATE['global_model'].set_weights(FL_STATE['current_global_weights'])
                app.logger.info(f"    Global model updated with FedAvg aggregation.")
            else:
                app.logger.warning("    FedAvg aggregation resulted in None weights. Global model not updated.")

            if current_round_client_sample_counts and sum(current_round_client_sample_counts) > 0:
                 avg_loss_this_round = np.average(current_round_client_losses, weights=current_round_client_sample_counts)
                 avg_acc_this_round = np.average(current_round_client_accuracies, weights=current_round_client_sample_counts)
            elif current_round_client_losses:
                 avg_loss_this_round = np.mean(current_round_client_losses)
                 avg_acc_this_round = np.mean(current_round_client_accuracies)


        else:
            FL_STATE['is_training_round_active'] = False
            app.logger.error(f"Unknown aggregation method: {config.aggregation_method}")
            raise ValueError(f"Método de agregação desconhecido: {config.aggregation_method}")

        FL_STATE['history_log']['round'].append(round_num)
        FL_STATE['history_log']['avg_client_loss'].append(avg_loss_this_round)
        FL_STATE['history_log']['avg_client_acc'].append(avg_acc_this_round)
        app.logger.info(f"    Average Client Loss (weighted): {avg_loss_this_round:.4f}, Average Client Accuracy (weighted): {avg_acc_this_round:.4f}")

        eval_frequency = getattr(config, 'eval_every', 1)
        test_loss, test_acc = float('nan'), float('nan')
        if round_num % eval_frequency == 0 or round_num == getattr(config, 'num_rounds_api_max', round_num):
            app.logger.info(f"  Evaluating global model on centralized test set for round {round_num}...")
            test_loss, test_acc = FL_STATE['global_model'].evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
            app.logger.info(f"  Global Model Evaluation (Round {round_num}): Loss={test_loss:.4f}, Accuracy={test_acc:.4f}")
        else:
            app.logger.debug(f"  Skipping global model evaluation for round {round_num} as eval_every is {eval_frequency}.")

        FL_STATE['history_log']['global_test_loss'].append(test_loss)
        FL_STATE['history_log']['global_test_acc'].append(test_acc)

        round_duration = time.time() - round_start_time
        app.logger.info(f"    Round Duration: {round_duration:.2f}s")

        FL_STATE['is_training_round_active'] = False
        return jsonify({
            "message": f"Rodada {round_num} concluída.",
            "round": round_num,
            "selected_clients_ns3_indices": sampled_original_client_indices,
            "avg_client_loss": avg_loss_this_round,
            "avg_client_accuracy": avg_acc_this_round,
            "global_test_loss": test_loss,
            "global_test_accuracy": test_acc,
            "round_duration_seconds": round_duration, # API internal round duration
            "simulated_client_performance": current_round_client_performance # New: detailed per-client results including sim time/size
        }), 200

    except Exception as e:
        app.logger.exception(f"Error in /run_round: {e}")
        FL_STATE['is_training_round_active'] = False
        if FL_STATE['history_log']['round'] and FL_STATE['history_log']['round'][-1] == FL_STATE['current_round']:
            for key in FL_STATE['history_log']: FL_STATE['history_log'][key].pop()
        FL_STATE['current_round'] = max(0, FL_STATE['current_round'] -1)
        return jsonify({"error": f"Erro interno ao executar rodada de treinamento: {str(e)}"}), 500

@app.route('/get_status', methods=['GET'])
def get_simulation_status():
    if FL_STATE['config'] is None:
        config_dict = "Não configurado"
    else:
        config_dict = vars(FL_STATE['config'])

    status = {
        "simulation_initialized": FL_STATE['simulation_initialized'],
        "data_loaded": FL_STATE['data_loaded'],
        "model_compiled": FL_STATE['model_compiled'],
        "current_round": FL_STATE['current_round'],
        "is_training_round_active": FL_STATE['is_training_round_active'],
        "configuration": config_dict,
        "num_total_clients": FL_STATE['config'].num_clients if FL_STATE['config'] else "N/A",
        "num_eligible_clients": len(FL_STATE['eligible_client_indices']) if FL_STATE['eligible_client_indices'] is not None else "N/A",
        "client_data_distribution (unique_samples)": FL_STATE['client_num_samples_unique'],
        "training_history": {
            "rounds": FL_STATE['history_log']['round'],
            "avg_client_loss": [f"{x:.4f}" if not np.isnan(x) else "NaN" for x in FL_STATE['history_log']['avg_client_loss']],
            "avg_client_accuracy": [f"{x:.4f}" if not np.isnan(x) else "NaN" for x in FL_STATE['history_log']['avg_client_acc']],
            "global_test_loss": [f"{x:.4f}" if not np.isnan(x) else "NaN" for x in FL_STATE['history_log']['global_test_loss']],
            "global_test_accuracy": [f"{x:.4f}" if not np.isnan(x) else "NaN" for x in FL_STATE['history_log']['global_test_acc']],
        }
    }
    app.logger.info(f"Status requested. Current round: {status['current_round']}, Initialized: {status['simulation_initialized']}")
    return jsonify(status), 200

@app.route('/evaluate_global_model', methods=['GET'])
def evaluate_global_model_endpoint():
    if not FL_STATE['simulation_initialized'] or not FL_STATE['global_model'] or not FL_STATE['centralized_test_dataset']:
        app.logger.warning("Attempted /evaluate_global_model while simulation/model/dataset not ready.")
        return jsonify({"error": "Simulação não inicializada ou modelo/dataset de teste não disponível."}), 400

    try:
        app.logger.info("Evaluating global model on demand...")
        loss, acc = FL_STATE['global_model'].evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
        app.logger.info(f"On-demand evaluation: Loss={loss:.4f}, Accuracy={acc:.4f}")
        return jsonify({"global_test_loss": loss, "global_test_accuracy": acc}), 200
    except Exception as e:
        app.logger.exception(f"Error in /evaluate_global_model: {e}")
        return jsonify({"error": f"Erro ao avaliar modelo: {str(e)}"}), 500

@app.route('/reset_simulation', methods=['POST'])
def reset_simulation():
    global FL_STATE
    if FL_STATE['is_training_round_active']:
         app.logger.warning("Attempted /reset_simulation while training round is active.")
         return jsonify({"error": "Não é possível resetar enquanto uma rodada de treinamento está ativa."}), 400

    FL_STATE = {
        "config": None, "client_train_datasets": None, "client_num_samples_unique": None,
        "centralized_test_dataset": None, "num_classes": None, "global_model": None,
        "current_global_weights": None, "eligible_client_indices": None,
        "history_log": collections.defaultdict(list), "current_round": 0,
        "simulation_initialized": False, "model_compiled": False, "data_loaded": False,
        "is_training_round_active": False
    }
    tf.keras.backend.clear_session()
    app.logger.info("FL simulation state reset.")
    return jsonify({"message": "Simulação resetada com sucesso."}), 200


@app.route('/get_client_info', methods=['GET'])
def get_client_info():
    if not FL_STATE['simulation_initialized'] or FL_STATE['client_num_samples_unique'] is None:
        app.logger.warning("Attempted /get_client_info while simulation/client data not ready.")
        return jsonify({"error": "Simulação não inicializada ou dados de cliente não carregados."}), 400

    client_id_str = request.args.get('client_id')

    if client_id_str:
        try:
            client_id = int(client_id_str)
            if not (0 <= client_id < FL_STATE['config'].num_clients):
                app.logger.error(f"Client ID {client_id} out of range in /get_client_info.")
                return jsonify({"error": f"client_id {client_id} fora do intervalo."}), 400

            info = {
                "client_id": client_id,
                "num_unique_samples": FL_STATE['client_num_samples_unique'][client_id],
                "is_eligible": client_id in FL_STATE['eligible_client_indices']
            }
            app.logger.debug(f"Returning info for client {client_id}: {info}")
            return jsonify(info), 200
        except ValueError:
            app.logger.error(f"Invalid client_id format: {client_id_str}")
            return jsonify({"error": "client_id deve ser um inteiro."}), 400
    else:
        all_clients_info = [
            {
                "client_id": i,
                "num_unique_samples": FL_STATE['client_num_samples_unique'][i],
                "is_eligible": i in FL_STATE['eligible_client_indices']
            } for i in range(FL_STATE['config'].num_clients)
        ]
        app.logger.debug(f"Returning info for all {len(all_clients_info)} clients.")
        return jsonify({"all_clients_info": all_clients_info}), 200


@app.route('/get_global_model_weights', methods=['GET'])
def get_global_model_weights_endpoint():
    if not FL_STATE['current_global_weights']:
        app.logger.warning("Attempted /get_global_model_weights while weights not available.")
        return jsonify({"error": "Pesos do modelo global não disponíveis."}), 400

    try:
        num_layers = len(FL_STATE['current_global_weights'])
        shapes = [w.shape for w in FL_STATE['current_global_weights']]
        app.logger.info(f"Returning metadata for global model weights. Layers: {num_layers}, Shapes: {shapes}")
        return jsonify({
            "message": "Pesos do modelo global recuperados (metadados).",
            "num_layers_with_weights": num_layers,
            "weight_shapes_per_layer": [str(s) for s in shapes]
        }), 200
    except Exception as e:
        app.logger.exception(f"Error in /get_global_model_weights: {e}")
        return jsonify({"error": f"Erro ao processar pesos: {str(e)}"}), 500


@app.route('/ping', methods=['GET'])
def ping():
    app.logger.info("Ping request received")
    return jsonify({"message": "pong"}), 200

if __name__ == '__main__':
    # Flask app will now use the handlers configured at the top of the file
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Starting port for the Flask API server')
    args, unknown = parser.parse_known_args() # Parse args, allow unknown ones for Flask/other libraries

    start_port = args.port
    port_range_end = start_port + 100 # Try up to 100 ports

    bound_port = None
    for port in range(start_port, port_range_end):
        try:
            # Check if the port is available by trying to bind a socket
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.bind(('127.0.0.1', port))
            temp_socket.close()
            # If bind successful, this port is likely available
            bound_port = port
            print(f"FL_API_PORT:{bound_port}") # Print the chosen port for the parent process (ns-3) to read
            break # Found an available port
        except OSError as e:
            if e.errno == 98: # Address already in use
                app.logger.warning(f"Port {port} already in use. Trying next port.")
            else:
                app.logger.error(f"Error checking port {port}: {e}")
        except Exception as e:
             app.logger.error(f"Unexpected error checking port {port}: {e}")


    if bound_port is None:
        app.logger.critical(f"Failed to find an available port in the range {start_port}-{port_range_end-1}.")
        exit(1) # Exit if no port is available

    app.logger.info(f"Starting Flask app on port {bound_port}")
    # Use the bound_port found in the loop
    app.run(debug=False, host='0.0.0.0', port=bound_port, threaded=False)
