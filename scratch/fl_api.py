import collections
import time
import os
import argparse # Usaremos para definir defaults
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from flask import Flask, request, jsonify
import threading # Para executar tarefas longas em segundo plano (opcional, mas bom)

# --- Configuração Inicial e Supressão de Logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

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
    return parser.parse_args([]) # Retorna args com defaults

# --- 1. Definição do Modelo Keras ---
# (Idêntica à original, mas usará FL_STATE['config'] implicitamente via outras funções)
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
    else: raise ValueError(f"Dataset {config.dataset} não suportado.")

    train_ds_global_tfds, ds_info = tfds.load(tfds_name, split='train', as_supervised=False, with_info=True, shuffle_files=True)
    test_ds_global_tfds = tfds.load(tfds_name, split='test', as_supervised=False)
    num_classes = ds_info.features['label'].num_classes
    total_train_samples_tfds = ds_info.splits['train'].num_examples

    print(f"Convertendo {total_train_samples_tfds} amostras de treino de TFDS para NumPy...")
    train_samples_np = list(tfds.as_numpy(train_ds_global_tfds))
    x_global_orig = np.array([s['image'] for s in train_samples_np])
    y_global_orig = np.array([s['label'] for s in train_samples_np])
    if y_global_orig.ndim > 1 and y_global_orig.shape[1] == 1:
        y_global_orig = y_global_orig.flatten()

    client_data_indices_map = [[] for _ in range(config.num_clients)]

    # A. Skew de Labels
    if config.non_iid_type == 'iid':
        all_indices = np.arange(total_train_samples_tfds)
        np.random.shuffle(all_indices)
        idx_splits = np.array_split(all_indices, config.num_clients)
        for i in range(config.num_clients): client_data_indices_map[i] = idx_splits[i].tolist()
    
    elif config.non_iid_type == 'pathological':
        num_classes_per_client = int(max(1, config.non_iid_alpha))
        print(f"  Particionamento Patológico: ~{num_classes_per_client} classes por cliente.")
        indices_by_label = [np.where(y_global_orig == i)[0] for i in range(num_classes)]
        for idx_list in indices_by_label: np.random.shuffle(idx_list)

        shards_per_label = []
        num_shards_target_per_label = max(1, (config.num_clients * num_classes_per_client) // num_classes)
        for label_idx_list in indices_by_label:
            if len(label_idx_list) > 0:
                shards_per_label.extend(np.array_split(label_idx_list, num_shards_target_per_label))
        np.random.shuffle(shards_per_label)

        # Técnica de atribuição de shards mais robusta
        client_shards_indices = [[] for _ in range(config.num_clients)]
        shards_assigned_count = 0
        # Distribui shards de forma round-robin para tentar equilibrar
        for i in range(len(shards_per_label)):
            client_idx_target = i % config.num_clients
            # Tenta garantir variedade de classes por cliente
            current_labels_for_client = set()
            if client_shards_indices[client_idx_target]:
                # Precisa recalcular labels se os índices já estão lá
                temp_indices = np.concatenate(client_shards_indices[client_idx_target])
                if len(temp_indices)>0:
                   current_labels_for_client = set(y_global_orig[temp_indices])


            # Se o cliente já tem o número desejado de classes únicas, pode pular para o próximo cliente
            # que ainda precisa, ou simplesmente continuar atribuindo para balancear a quantidade de shards
            # Esta lógica pode ser complexa. Simplificando: atribuir até num_classes_per_client shards.
            if len(client_shards_indices[client_idx_target]) < num_classes_per_client :
                 shard_to_assign = shards_per_label[i]
                 client_shards_indices[client_idx_target].append(shard_to_assign)

        # Se alguns clientes ficaram com poucos shards, redistribui
        # (Esta parte da lógica original é um pouco complexa de replicar fielmente sem depuração profunda)
        # Simplificando: cada cliente pega num_classes_per_client shards distintos se possível
        
        client_labels = [set() for _ in range(config.num_clients)]
        shard_ptr = 0
        for client_id in range(config.num_clients):
            for _ in range(num_classes_per_client):
                if shard_ptr >= len(shards_per_label): break # Se acabaram os shards
                
                # Procura um shard de uma classe que o cliente ainda não tenha
                found_new_label_shard = False
                initial_shard_ptr = shard_ptr
                while True:
                    current_shard = shards_per_label[shard_ptr % len(shards_per_label)]
                    if len(current_shard) > 0:
                        label_of_shard = y_global_orig[current_shard[0]]
                        if label_of_shard not in client_labels[client_id]:
                            client_data_indices_map[client_id].extend(current_shard.tolist())
                            client_labels[client_id].add(label_of_shard)
                            # Marcar shard como usado (ou remover, mas pode complicar a indexação circular)
                            # Para simplificar, vamos apenas avançar o ponteiro. Pode haver reutilização
                            # se num_classes_per_client * num_clients > num_shards.
                            found_new_label_shard = True
                            shard_ptr += 1
                            break
                    shard_ptr += 1
                    if (shard_ptr % len(shards_per_label)) == (initial_shard_ptr % len(shards_per_label)):
                        # Deu a volta completa e não achou shard de nova classe
                        break 
                if not found_new_label_shard and shard_ptr < len(shards_per_label) : # Se não achou novo, pega qualquer um
                    current_shard = shards_per_label[shard_ptr % len(shards_per_label)]
                    if len(current_shard) > 0:
                         client_data_indices_map[client_id].extend(current_shard.tolist())
                    shard_ptr+=1

            if client_data_indices_map[client_id]:
                client_data_indices_map[client_id] = list(np.unique(np.array(client_data_indices_map[client_id], dtype=int)))


    elif config.non_iid_type == 'dirichlet':
        print(f"  Particionamento Dirichlet com alpha={config.non_iid_alpha}.")
        label_proportions = np.random.dirichlet([config.non_iid_alpha] * num_classes, config.num_clients)
        indices_by_label = [np.where(y_global_orig == i)[0] for i in range(num_classes)]
        for idx_list in indices_by_label: np.random.shuffle(idx_list) # Embaralha dentro de cada classe
        
        ptr_by_label = [0] * num_classes

        for client_idx in range(config.num_clients):
            client_indices_list = []
            # Determina o número total de amostras para este cliente (poderia ser melhorado para respeitar o total)
            # Uma abordagem simples: dividir o total de amostras igualmente e aplicar proporções dirichlet a isso.
            # Ou, como no original, aplicar proporção ao total de cada classe e distribuir.
            # O original é um pouco complexo porque pode não usar todas as amostras.
            # Tentativa de seguir o espírito do original:
            
            # Para cada cliente, calculamos quantas amostras de cada classe ele "quer"
            # Esta parte precisa ser cuidadosa para não atribuir mais amostras do que existem
            # e para distribuir todas as amostras se possível.
            
            # Alternativa mais simples (e comum) para Dirichlet:
            # 1. Para cada classe, dividir os seus índices de acordo com as proporções de Dirichlet para aquela classe entre os clientes.
            # Esta é mais difícil de implementar corretamente para garantir que todos os dados sejam usados.
            # A abordagem no código original (iterar clientes, depois classes) é mais comum:
            for label_k in range(num_classes):
                indices_for_label_k = indices_by_label[label_k]
                num_samples_for_label_k = len(indices_for_label_k)
                if num_samples_for_label_k == 0: continue

                # Quantas amostras o cliente atual "quer" desta classe
                # Isso é uma proporção das *amostras totais disponíveis para esta classe*
                # E não do total de amostras do cliente.
                
                # Vamos calcular a quantidade alvo de amostras por cliente por classe
                # A soma das proporções de Dirichlet para uma classe (coluna) não é 1.
                # A soma das proporções de Dirichlet para um cliente (linha) é 1.
                # Então, label_proportions[client_idx, label_k] é a fração de dados *deste cliente* que pertence à classe k.

                # Precisamos de uma estimativa de amostras por cliente. Se for distribuição uniforme de quantidade total:
                # approx_samples_per_client = total_train_samples_tfds // config.num_clients
                # num_to_take_from_label_k = int(label_proportions[client_idx, label_k] * approx_samples_per_client)
                
                # O código original parece fazer o seguinte (que é mais complexo de balancear globalmente):
                # Para cada cliente, decide quantos de cada classe pegar baseado na proporção E no total da classe.
                # Esta parte é tricky, pois o total de "num_to_take" pode exceder o disponível.
                # Tentando reproduzir a lógica original:
                
                target_num_samples_from_class_for_client = int(np.round(label_proportions[client_idx, label_k] * (total_train_samples_tfds / config.num_clients)))
                
                start_ptr = ptr_by_label[label_k]
                num_available_for_label_k = num_samples_for_label_k - start_ptr
                
                num_actually_taken = min(target_num_samples_from_class_for_client, num_available_for_label_k)

                if num_actually_taken > 0:
                    end_ptr = start_ptr + num_actually_taken
                    client_indices_list.extend(indices_by_label[label_k][start_ptr:end_ptr])
                    ptr_by_label[label_k] = end_ptr
            
            np.random.shuffle(client_indices_list)
            client_data_indices_map[client_idx] = client_indices_list
    else:
        raise ValueError(f"Tipo de não-IID desconhecido: {config.non_iid_type}")

    # B. Aplicar Skew de Quantidade
    if config.quantity_skew_type == 'power_law':
        print(f"  Aplicando Power Law Quantity Skew com beta={config.power_law_beta}.")
        min_samples_per_client = max(1, config.batch_size // 4)
        raw_power_law_samples = np.random.pareto(config.power_law_beta, config.num_clients) + 1e-6
        proportions = raw_power_law_samples / np.sum(raw_power_law_samples)

        # total_samples_currently_assigned = sum(len(idx_list) for idx_list in client_data_indices_map)
        # Se quisermos que a soma das amostras *após* power law seja igual a total_train_samples_tfds:
        # (isso pode ser muito restritivo se o label skew já removeu muitas amostras)
        # Usaremos o total de amostras *atualmente atribuídas* como base para o power law.
        
        total_samples_to_distribute_by_power_law = sum(len(indices) for indices in client_data_indices_map)
        target_samples_per_client = (proportions * total_samples_to_distribute_by_power_law).astype(int)
        target_samples_per_client = np.maximum(target_samples_per_client, min_samples_per_client)

        # Ajustar para que a soma não exceda o total disponível (reduz proporcionalmente)
        # Este passo é importante se 'total_samples_to_distribute_by_power_law' for o total global original.
        # Se for baseado no que já foi atribuído, pode ser menos crítico.
        current_sum_target = np.sum(target_samples_per_client)
        if current_sum_target > total_samples_to_distribute_by_power_law and total_samples_to_distribute_by_power_law > 0:
            excess = current_sum_target - total_samples_to_distribute_by_power_law
            reduction_factors = target_samples_per_client / current_sum_target
            reductions = (reduction_factors * excess).astype(int)
            target_samples_per_client = np.maximum(min_samples_per_client, target_samples_per_client - reductions)
        elif total_samples_to_distribute_by_power_law == 0: # Caso extremo
             target_samples_per_client = np.zeros_like(target_samples_per_client)


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

    # C. Criar tf.data.Datasets
    client_datasets_train_list = []
    client_num_samples_unique_list = []

    for c_idx in range(config.num_clients):
        indices_for_this_client = np.array(client_data_indices_map[c_idx], dtype=int)
        num_unique_samples = len(indices_for_this_client)
        client_num_samples_unique_list.append(num_unique_samples)

        if num_unique_samples == 0:
            empty_x = np.array([], dtype=x_global_orig.dtype).reshape(0, *x_global_orig.shape[1:])
            empty_y = np.array([], dtype=y_global_orig.dtype)
            client_tf_ds = create_client_tf_dataset_api(empty_x, empty_y, config)
            client_datasets_train_list.append(client_tf_ds)
            continue

        client_x_data_orig = x_global_orig[indices_for_this_client]
        client_y_data_orig = y_global_orig[indices_for_this_client]

        client_x_processed = client_x_data_orig.astype('float32')
        if config.feature_skew_type == 'noise' and config.noise_std_dev > 0:
            noise_abs_std_dev = config.noise_std_dev * 255.0
            noise = np.random.normal(0, noise_abs_std_dev, client_x_processed.shape).astype(client_x_processed.dtype)
            client_x_processed = client_x_processed + noise
            client_x_processed = np.clip(client_x_processed, 0.0, 255.0)
        
        client_tf_ds = create_client_tf_dataset_api(client_x_processed, client_y_data_orig, config)
        client_datasets_train_list.append(client_tf_ds)
    
    centralized_test_data = test_ds_global_tfds.map(preprocess_dataset_for_model_creation_api, num_parallel_calls=tf.data.AUTOTUNE)\
                                               .batch(config.batch_size * 2)\
                                               .prefetch(tf.data.AUTOTUNE)

    print(f"Distribuição de dados concluída. Amostras por cliente: {client_num_samples_unique_list}")
    
    FL_STATE['client_train_datasets'] = client_datasets_train_list
    FL_STATE['client_num_samples_unique'] = client_num_samples_unique_list
    FL_STATE['centralized_test_dataset'] = centralized_test_data
    FL_STATE['num_classes'] = num_classes
    FL_STATE['eligible_client_indices'] = [i for i, n_samples in enumerate(client_num_samples_unique_list) if n_samples > 0]
    FL_STATE['data_loaded'] = True
    
    return True # Success

# --- 3. Treinamento e Agregação Federada (adaptado para usar FL_STATE['config']) ---
def client_update_api(model_template, global_weights, client_tf_dataset, num_unique_samples_this_client):
    config = FL_STATE['config']
    if num_unique_samples_this_client == 0:
        return global_weights, 0.0, 0.0

    client_model = tf.keras.models.clone_model(model_template) # Clona a estrutura
    # Precisa compilar o clone antes de setar pesos se o modelo original não foi compilado
    # ou se o otimizador/métricas são diferentes. Aqui, vamos compilar sempre para segurança.
    
    if config.client_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=config.client_lr)
    elif config.client_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.client_lr)
    else:
        raise ValueError(f"Otimizador de cliente desconhecido: {config.client_optimizer}")

    client_model.compile(optimizer=optimizer,
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    client_model.set_weights(global_weights) # Agora define os pesos

    steps_per_epoch_val = None
    if num_unique_samples_this_client > 0:
         steps_per_epoch_val = int(np.ceil(num_unique_samples_this_client * config.local_epochs / config.batch_size))

    # client_tf_dataset já está com repeat(local_epochs)
    # então epochs=1 no fit.
    history = client_model.fit(client_tf_dataset, epochs=1, verbose=0, steps_per_epoch=steps_per_epoch_val)
    
    loss = history.history['loss'][-1] if 'loss' in history.history and history.history['loss'] else 0.0
    accuracy = history.history['sparse_categorical_accuracy'][-1] if 'sparse_categorical_accuracy' in history.history and history.history['sparse_categorical_accuracy'] else 0.0
    return client_model.get_weights(), loss, accuracy

def aggregate_weights_fedavg_api(client_weights_list, client_num_samples_list):
    if not client_weights_list: return None
    total_samples_in_round = sum(client_num_samples_list)
    if total_samples_in_round == 0:
        print("  Aviso: Nenhum dado de cliente para agregar nesta rodada.")
        return client_weights_list[0] if client_weights_list else None

    avg_weights = [np.zeros_like(w) for w in client_weights_list[0]]
    for i, client_weights in enumerate(client_weights_list):
        num_samples = client_num_samples_list[i]
        if num_samples == 0: continue
        weight_factor = num_samples / total_samples_in_round
        for layer_idx, layer_weights in enumerate(client_weights):
            avg_weights[layer_idx] += weight_factor * layer_weights
    return avg_weights

# --- Endpoints da API Flask ---

@app.route('/configure', methods=['POST'])
def configure_simulation():
    global FL_STATE
    if FL_STATE['simulation_initialized'] or FL_STATE['is_training_round_active']:
        return jsonify({"error": "A simulação já está em andamento ou inicializada. Use /reset_simulation primeiro."}), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Payload JSON não fornecido ou inválido."}), 400

        # Começa com defaults e sobrescreve com o que foi passado
        current_args = get_default_args()
        # Valida e atualiza os argumentos
        for key, value in data.items():
            if hasattr(current_args, key):
                # Tenta converter para o tipo esperado, se possível
                # Isso é uma validação básica, idealmente usar Marshmallow ou Pydantic
                default_val = getattr(current_args, key)
                if default_val is not None:
                    try:
                        setattr(current_args, key, type(default_val)(value))
                    except (ValueError, TypeError) as e:
                         return jsonify({"error": f"Valor inválido para '{key}': {value}. Esperado tipo {type(default_val)}. Erro: {e}"}), 400
                else: # Se o default é None, aceita o tipo que veio
                    setattr(current_args, key, value)
            else:
                app.logger.warning(f"Chave de configuração desconhecida '{key}' ignorada.")
        
        FL_STATE['config'] = current_args
        FL_STATE['simulation_initialized'] = False # Resetar flags dependentes
        FL_STATE['model_compiled'] = False
        FL_STATE['data_loaded'] = False
        FL_STATE['current_round'] = 0
        FL_STATE['history_log'] = collections.defaultdict(list)
        
        return jsonify({"message": "Configuração recebida com sucesso.", "config": vars(FL_STATE['config'])}), 200
    except Exception as e:
        app.logger.error(f"Erro em /configure: {e}", exc_info=True)
        return jsonify({"error": f"Erro interno ao processar configuração: {str(e)}"}), 500

@app.route('/initialize_simulation', methods=['POST'])
def initialize_simulation():
    global FL_STATE
    if FL_STATE['is_training_round_active']:
        return jsonify({"error": "Uma rodada de treinamento está ativa."}), 400
    if FL_STATE['simulation_initialized']:
        return jsonify({"message": "Simulação já inicializada. Use /reset_simulation para reconfigurar."}), 200
    if FL_STATE['config'] is None:
        return jsonify({"error": "Configuração não definida. Chame /configure primeiro."}), 400

    try:
        # Definir sementes globais
        if FL_STATE['config'].seed is not None:
            tf.keras.utils.set_random_seed(FL_STATE['config'].seed)
            np.random.seed(FL_STATE['config'].seed) # Redundante se set_random_seed já faz, mas garante

        # 1. Carregar e distribuir dados
        print("Inicializando: Carregando e distribuindo dados...")
        data_load_success = load_and_distribute_data_api()
        if not data_load_success: # Deveria levantar exceção em caso de falha, mas por via das dúvidas
            return jsonify({"error": "Falha ao carregar/distribuir dados."}), 500
        print(f"Dados carregados. Clientes elegíveis: {FL_STATE['eligible_client_indices']}")

        # 2. Criar e compilar modelo global
        print("Inicializando: Criando modelo global...")
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
        print("Modelo global criado e compilado.")

        FL_STATE['simulation_initialized'] = True
        FL_STATE['current_round'] = 0
        FL_STATE['history_log'] = collections.defaultdict(list) # Limpa histórico
        
        # Avaliação inicial do modelo (opcional, mas bom para ter um baseline)
        if FL_STATE['centralized_test_dataset'] and FL_STATE['global_model']:
            print("Avaliando modelo global inicial...")
            initial_loss, initial_acc = FL_STATE['global_model'].evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
            FL_STATE['history_log']['round'].append(0)
            FL_STATE['history_log']['avg_client_loss'].append(float('nan'))
            FL_STATE['history_log']['avg_client_acc'].append(float('nan'))
            FL_STATE['history_log']['global_test_loss'].append(initial_loss)
            FL_STATE['history_log']['global_test_acc'].append(initial_acc)
            print(f"Modelo Inicial (Rodada 0) - Perda Teste: {initial_loss:.4f}, Acurácia Teste: {initial_acc:.4f}")
        
        return jsonify({
            "message": "Simulação FL inicializada com sucesso.",
            "num_clients_total": FL_STATE['config'].num_clients,
            "num_eligible_clients": len(FL_STATE['eligible_client_indices']),
            "client_samples_distribution": FL_STATE['client_num_samples_unique'],
            "initial_model_evaluation": {
                "loss": FL_STATE['history_log']['global_test_loss'][-1] if FL_STATE['history_log']['global_test_loss'] else 'N/A',
                "accuracy": FL_STATE['history_log']['global_test_acc'][-1] if FL_STATE['history_log']['global_test_acc'] else 'N/A'
            }
        }), 200

    except Exception as e:
        app.logger.error(f"Erro em /initialize_simulation: {e}", exc_info=True)
        FL_STATE['simulation_initialized'] = False # Reverte em caso de erro
        return jsonify({"error": f"Erro interno ao inicializar simulação: {str(e)}"}), 500

@app.route('/run_round', methods=['POST'])
def run_training_round():
    global FL_STATE
    if FL_STATE['is_training_round_active']:
        return jsonify({"error": "Outra rodada de treinamento já está em andamento."}), 409 # Conflict
    if not FL_STATE['simulation_initialized']:
        return jsonify({"error": "Simulação não inicializada. Chame /initialize_simulation primeiro."}), 400
    if not FL_STATE['eligible_client_indices']:
        return jsonify({"error": "Nenhum cliente elegível com dados para treinamento."}), 400

    FL_STATE['is_training_round_active'] = True
    try:
        config = FL_STATE['config']
        FL_STATE['current_round'] += 1
        round_num = FL_STATE['current_round']
        
        print(f"\n--- Iniciando Rodada de Treinamento FL: {round_num} ---")
        round_start_time = time.time()

        data = request.get_json(silent=True) or {}
        selected_client_indices_input = data.get('client_indices', None)
        
        sampled_original_client_indices = []
        if selected_client_indices_input is not None:
            if not isinstance(selected_client_indices_input, list) or not all(isinstance(i, int) for i in selected_client_indices_input):
                FL_STATE['is_training_round_active'] = False
                return jsonify({"error": "client_indices deve ser uma lista de inteiros."}), 400
            
            # Valida se os índices fornecidos são elegíveis
            sampled_original_client_indices = [
                idx for idx in selected_client_indices_input if idx in FL_STATE['eligible_client_indices']
            ]
            if len(sampled_original_client_indices) != len(selected_client_indices_input):
                app.logger.warning(f"Alguns clientes selecionados manualmente não são elegíveis ou não existem. Usando apenas os válidos.")
            if not sampled_original_client_indices:
                FL_STATE['is_training_round_active'] = False
                return jsonify({"error": "Nenhum dos clientes selecionados manualmente é elegível ou possui dados."}), 400
        else:
            num_clients_to_sample = min(config.clients_per_round, len(FL_STATE['eligible_client_indices']))
            if num_clients_to_sample > 0:
                sampled_original_client_indices = np.random.choice(
                    FL_STATE['eligible_client_indices'],
                    size=num_clients_to_sample,
                    replace=False
                ).tolist()
            else: # Não deveria acontecer se eligible_client_indices tem itens
                 FL_STATE['is_training_round_active'] = False
                 return jsonify({"message": "Nenhum cliente para amostrar nesta rodada (verifique clients_per_round e dados dos clientes)."}), 200


        print(f"  Rodada {round_num}/{config.num_rounds_api_max if hasattr(config, 'num_rounds_api_max') else 'N/A'} - Clientes amostrados (índices originais): {sampled_original_client_indices}")

        current_round_client_updates = []
        current_round_client_sample_counts = []
        current_round_client_losses = []
        current_round_client_accuracies = []

        for client_original_idx in sampled_original_client_indices:
            client_ds_for_training = FL_STATE['client_train_datasets'][client_original_idx]
            num_unique_samples = FL_STATE['client_num_samples_unique'][client_original_idx]

            if num_unique_samples == 0: # Segurança, já filtrado por eligible_client_indices
                continue

            print(f"    Treinando cliente {client_original_idx} com {num_unique_samples} amostras...")
            updated_w, loss, acc = client_update_api(
                FL_STATE['global_model'], # Passa o modelo template (estrutura)
                FL_STATE['current_global_weights'],
                client_ds_for_training,
                num_unique_samples
            )
            current_round_client_updates.append(updated_w)
            current_round_client_sample_counts.append(num_unique_samples)
            current_round_client_losses.append(loss)
            current_round_client_accuracies.append(acc)
            print(f"      Cliente {client_original_idx} - Perda: {loss:.4f}, Acurácia: {acc:.4f}")

        avg_loss_this_round = float('nan')
        avg_acc_this_round = float('nan')

        if not current_round_client_updates:
            print(f"    Nenhum cliente treinou nesta rodada. Mantendo pesos globais.")
        elif config.aggregation_method == 'fedavg':
            new_global_weights = aggregate_weights_fedavg_api(current_round_client_updates, current_round_client_sample_counts)
            if new_global_weights is not None:
                FL_STATE['current_global_weights'] = new_global_weights
                FL_STATE['global_model'].set_weights(FL_STATE['current_global_weights']) # Atualiza o modelo global
            
            if current_round_client_sample_counts and sum(current_round_client_sample_counts) > 0:
                 avg_loss_this_round = np.average(current_round_client_losses, weights=current_round_client_sample_counts)
                 avg_acc_this_round = np.average(current_round_client_accuracies, weights=current_round_client_sample_counts)
            elif current_round_client_losses : # Caso todos os clientes treinados tenham 0 amostras (improvável com a lógica atual)
                 avg_loss_this_round = np.mean(current_round_client_losses)
                 avg_acc_this_round = np.mean(current_round_client_accuracies)

        else:
            FL_STATE['is_training_round_active'] = False
            raise ValueError(f"Método de agregação desconhecido: {config.aggregation_method}")

        FL_STATE['history_log']['round'].append(round_num)
        FL_STATE['history_log']['avg_client_loss'].append(avg_loss_this_round)
        FL_STATE['history_log']['avg_client_acc'].append(avg_acc_this_round)
        print(f"    Perda Média Cliente (pond.): {avg_loss_this_round:.4f}, Acurácia Média Cliente (pond.): {avg_acc_this_round:.4f}")

        eval_frequency = getattr(config, 'eval_every', 1) # Default para eval_every se não estiver no config
        test_loss, test_acc = float('nan'), float('nan')
        if round_num % eval_frequency == 0 or round_num == getattr(config, 'num_rounds_api_max', round_num): # Adicionado getattr para num_rounds_api_max
            print(f"  Avaliando modelo global na rodada {round_num}...")
            test_loss, test_acc = FL_STATE['global_model'].evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
            print(f"  Avaliação Global (Rodada {round_num}): Perda={test_loss:.4f}, Acurácia={test_acc:.4f}")
        
        FL_STATE['history_log']['global_test_loss'].append(test_loss)
        FL_STATE['history_log']['global_test_acc'].append(test_acc)
        
        round_duration = time.time() - round_start_time
        print(f"    Tempo da Rodada: {round_duration:.2f}s")
        
        FL_STATE['is_training_round_active'] = False
        return jsonify({
            "message": f"Rodada {round_num} concluída.",
            "round": round_num,
            "selected_clients": sampled_original_client_indices,
            "avg_client_loss": avg_loss_this_round,
            "avg_client_accuracy": avg_acc_this_round,
            "global_test_loss": test_loss,
            "global_test_accuracy": test_acc,
            "round_duration_seconds": round_duration
        }), 200

    except Exception as e:
        app.logger.error(f"Erro em /run_round: {e}", exc_info=True)
        FL_STATE['is_training_round_active'] = False
        # Tenta reverter o incremento da rodada se houve erro
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
        # Remover objetos TFDS da resposta JSON para evitar problemas de serialização
        # config_dict = {k: v for k, v in config_dict.items() if not isinstance(v, tf.data.Dataset)}


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
    return jsonify(status), 200

@app.route('/evaluate_global_model', methods=['GET'])
def evaluate_global_model_endpoint():
    if not FL_STATE['simulation_initialized'] or not FL_STATE['global_model'] or not FL_STATE['centralized_test_dataset']:
        return jsonify({"error": "Simulação não inicializada ou modelo/dataset de teste não disponível."}), 400
    
    try:
        print("Avaliando modelo global sob demanda...")
        loss, acc = FL_STATE['global_model'].evaluate(FL_STATE['centralized_test_dataset'], verbose=0)
        print(f"Avaliação sob demanda: Perda={loss:.4f}, Acurácia={acc:.4f}")
        return jsonify({"global_test_loss": loss, "global_test_accuracy": acc}), 200
    except Exception as e:
        app.logger.error(f"Erro em /evaluate_global_model: {e}", exc_info=True)
        return jsonify({"error": f"Erro ao avaliar modelo: {str(e)}"}), 500

@app.route('/reset_simulation', methods=['POST'])
def reset_simulation():
    global FL_STATE
    if FL_STATE['is_training_round_active']:
         return jsonify({"error": "Não é possível resetar enquanto uma rodada de treinamento está ativa."}), 400
    
    FL_STATE = {
        "config": None, "client_train_datasets": None, "client_num_samples_unique": None,
        "centralized_test_dataset": None, "num_classes": None, "global_model": None,
        "current_global_weights": None, "eligible_client_indices": None,
        "history_log": collections.defaultdict(list), "current_round": 0,
        "simulation_initialized": False, "model_compiled": False, "data_loaded": False,
        "is_training_round_active": False
    }
    # Limpar sessão Keras pode ser útil para liberar memória de modelos anteriores
    tf.keras.backend.clear_session()
    print("Estado da simulação FL resetado.")
    return jsonify({"message": "Simulação resetada com sucesso."}), 200


@app.route('/get_client_info', methods=['GET'])
def get_client_info():
    if not FL_STATE['simulation_initialized'] or FL_STATE['client_num_samples_unique'] is None:
        return jsonify({"error": "Simulação não inicializada ou dados de cliente não carregados."}), 400

    client_id_str = request.args.get('client_id')
    
    if client_id_str:
        try:
            client_id = int(client_id_str)
            if not (0 <= client_id < FL_STATE['config'].num_clients):
                return jsonify({"error": f"client_id {client_id} fora do intervalo."}), 400
            
            info = {
                "client_id": client_id,
                "num_unique_samples": FL_STATE['client_num_samples_unique'][client_id],
                "is_eligible": client_id in FL_STATE['eligible_client_indices']
                # Poderia adicionar mais info, ex: distribuição de labels se pré-calculada
            }
            return jsonify(info), 200
        except ValueError:
            return jsonify({"error": "client_id deve ser um inteiro."}), 400
    else:
        all_clients_info = [
            {
                "client_id": i,
                "num_unique_samples": FL_STATE['client_num_samples_unique'][i],
                "is_eligible": i in FL_STATE['eligible_client_indices']
            } for i in range(FL_STATE['config'].num_clients)
        ]
        return jsonify({"all_clients_info": all_clients_info}), 200


# Funcionalidades adicionais (não solicitadas diretamente, mas úteis)
@app.route('/get_global_model_weights', methods=['GET'])
def get_global_model_weights_endpoint():
    if not FL_STATE['current_global_weights']:
        return jsonify({"error": "Pesos do modelo global não disponíveis."}), 400
    
    try:
        # Serializar pesos para JSON é complicado. Melhor salvar em disco e retornar um path,
        # ou uma representação textual simples se for apenas para inspeção leve.
        # Para este exemplo, vamos apenas confirmar que existem.
        # Para uma API real, você pode usar np.save para um arquivo temporário e permitir o download,
        # ou converter para listas (pode ser muito grande).
        num_layers = len(FL_STATE['current_global_weights'])
        shapes = [w.shape for w in FL_STATE['current_global_weights']]
        return jsonify({
            "message": "Pesos do modelo global recuperados (metadados).",
            "num_layers_with_weights": num_layers,
            "weight_shapes_per_layer": [str(s) for s in shapes] # Convertendo shapes para string
        }), 200
    except Exception as e:
        app.logger.error(f"Erro em /get_global_model_weights: {e}", exc_info=True)
        return jsonify({"error": f"Erro ao processar pesos: {str(e)}"}), 500


@app.route('/ping', methods=['GET'])
def ping():
    app.logger.info("Ping request received")
    return jsonify({"message": "pong"}), 200


# Não implementaremos /set_global_model_weights por simplicidade, pois envolveria
# upload de arquivos de pesos e recriação/validação cuidadosa do modelo.

if __name__ == '__main__':
    # Configurar logging do Flask para melhor visibilidade
    import logging
    logging.basicConfig(level=logging.INFO)
    handler = logging.StreamHandler() # Para sair no console
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    # Nota: O servidor de desenvolvimento Flask não é ideal para produção.
    # Use Gunicorn ou uWSGI por trás de um proxy reverso como Nginx.
    # Para TensorFlow, pode ser necessário executar com `threaded=False` se houver problemas de grafo,
    # ou garantir que cada request tenha seu próprio grafo (mais complexo).
    # Por padrão, o servidor de dev do Flask é single-threaded.
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=False) # threaded=True para dev