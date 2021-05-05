"""
    Created by:
        Jay Lago - 17 Nov 2020
"""
import tensorflow as tf
import numpy as np
import pickle
import datetime as dt

import KoopmanMachine as km
import LossFunction as lf
import Data as dat
import Training as tr


# ==============================================================================
# Setup
# ==============================================================================
NEW_DATA = True     # Keep this flag True if running script for the first time
NUM_SAVES = 1       # Number of times to save the model while training
NUM_PLOTS = 20      # Number of diagnostic plots to generate while training
DEVICE = '/GPU:0'
GPUS = tf.config.experimental.list_physical_devices('GPU')
if GPUS:
    try:
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    DEVICE = '/CPU:0'

tf.keras.backend.set_floatx('float64')  # !! Set precision for the entire model here
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Num GPUs Available: {}".format(len(GPUS)))
print("Training at precision: {}".format(tf.keras.backend.floatx()))
print("Training on device: {}".format(DEVICE))


# ==============================================================================
# Initialize hyper-parameters and Koopman model
# ==============================================================================
# General parameters
hyp_params = dict()
hyp_params['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp_params['experiment'] = 'fluid_flow_slow'
hyp_params['plot_path'] = './training/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']
hyp_params['model_path'] = './models/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']
hyp_params['device'] = DEVICE
hyp_params['precision'] = tf.keras.backend.floatx()
hyp_params['num_init_conds'] = 15000
hyp_params['num_train_init_conds'] = int(0.7 * hyp_params['num_init_conds'])
hyp_params['num_test_init_conds'] = int(0.3 * hyp_params['num_init_conds'])
hyp_params['time_final'] = 6
hyp_params['delta_t'] = 0.05
hyp_params['num_time_steps'] = int(hyp_params['time_final'] / hyp_params['delta_t'] + 1)
hyp_params['num_pred_steps'] = 30
hyp_params['max_epochs'] = 100
hyp_params['save_every'] = hyp_params['max_epochs'] // NUM_SAVES
hyp_params['plot_every'] = hyp_params['max_epochs'] // NUM_PLOTS
hyp_params['pretrain'] = False

# Universal network layer parameters (AE & Aux)
hyp_params['optimizer'] = 'adam'
hyp_params['batch_size'] = 512
hyp_params['phys_dim'] = 3
hyp_params['num_cmplx_prs'] = 1
hyp_params['num_real'] = 0
hyp_params['latent_dim'] = 2*hyp_params['num_cmplx_prs'] + hyp_params['num_real']
hyp_params['hidden_activation'] = tf.keras.activations.relu
hyp_params['bias_initializer'] = tf.keras.initializers.Zeros

# Encoding/Decoding Layer Parameters
hyp_params['num_en_layers'] = 3
hyp_params['num_en_neurons'] = 100
hyp_params['kernel_init_enc'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['kernel_init_dec'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['ae_output_activation'] = tf.keras.activations.linear

# Auxiliary Layer Parameters
hyp_params['num_k_layers'] = 3
hyp_params['num_k_neurons'] = 300
hyp_params['kernel_init_aux'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['aux_output_activation'] = tf.keras.activations.linear

# Loss Function Parameters
hyp_params['a1'] = tf.constant(1e-1, dtype=hyp_params['precision'])  # Reconstruction
hyp_params['a2'] = tf.constant(1, dtype=hyp_params['precision'])  # Prediction
hyp_params['a3'] = tf.constant(1, dtype=hyp_params['precision'])  # Linearity
hyp_params['a4'] = tf.constant(1e-7, dtype=hyp_params['precision'])  # L-inf
hyp_params['a5'] = tf.constant(1e-13, dtype=hyp_params['precision'])  # L-2 on weights

# Learning rate
hyp_params['lr'] = 1e-3

# Initialize Koopman machine and loss
myMachine = km.KoopmanMachine(hyp_params)
myLoss = lf.LossFunction(hyp_params)


# ==============================================================================
# Generate / load data
# ==============================================================================
data_path = './data/dataset_fluid_slow.pkl'
if NEW_DATA:
    data = dat.data_maker_fluid_flow_slow(r_lower=0, r_upper=1.1, t_lower=0, t_upper=2*np.pi,
                                          n_ic=hyp_params['num_init_conds'], dt=hyp_params['delta_t'],
                                          tf=hyp_params['time_final'])
    data = tf.cast(data, dtype=hyp_params['precision'])
    # Save data to file
    pickle.dump(data, open(data_path, 'wb'))
else:
    # Load data from file
    data = pickle.load(open(data_path, 'rb'))
    data = tf.cast(data, dtype=hyp_params['precision'])

# Create training and validation datasets from the initial conditions
shuffled_data = tf.random.shuffle(data)
train_data = tf.data.Dataset.from_tensor_slices(shuffled_data[:hyp_params['num_train_init_conds'], :, :])
val_data = tf.data.Dataset.from_tensor_slices(shuffled_data[hyp_params['num_train_init_conds']:, :, :])

# Batch and prefetch the validation data to the GPUs
val_set = val_data.batch(hyp_params['batch_size'], drop_remainder=True)
val_set = val_set.prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# Train model
# ==============================================================================
results = tr.train_model(hyp_params=hyp_params, train_data=train_data,
                         val_set=val_set, model=myMachine, loss=myLoss)

print(results['model'].summary())
exit()
