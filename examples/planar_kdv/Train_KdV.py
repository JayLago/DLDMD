"""
    Created by:
        Jay Lago - 20 May 2021
"""
import tensorflow as tf
import pickle
import datetime as dt
import os

import DLDMD as dl
import LossDLDMD as lf
import Data as dat
import Training as tr


# ==============================================================================
# Setup
# ==============================================================================
NUM_SAVES = 1       # Number of times to save the model throughout training
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
# Initialize hyper-parameters and model
# ==============================================================================
# General parameters
hyp_params = dict()
hyp_params['sim_start'] = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
hyp_params['experiment'] = 'kdv'
hyp_params['plot_path'] = './training_results/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']
hyp_params['model_path'] = './trained_models/' + hyp_params['experiment'] + '_' + hyp_params['sim_start']
hyp_params['device'] = DEVICE
hyp_params['precision'] = tf.keras.backend.floatx()
hyp_params['num_init_conds'] = 10000
hyp_params['num_train_init_conds'] = 8000
hyp_params['num_val_init_conds'] = 2000
hyp_params['time_final'] = 6
hyp_params['delta_t'] = 0.02
hyp_params['num_time_steps'] = int(hyp_params['time_final']/hyp_params['delta_t'] + 1)
hyp_params['num_pred_steps'] = hyp_params['num_time_steps']
hyp_params['max_epochs'] = 100
hyp_params['save_every'] = hyp_params['max_epochs'] // NUM_SAVES
hyp_params['plot_every'] = hyp_params['max_epochs'] // NUM_PLOTS
hyp_params['pretrain'] = False

# Universal network layer parameters (AE & Aux)
hyp_params['optimizer'] = 'adam'
hyp_params['batch_size'] = 256
hyp_params['phys_dim'] = 2
hyp_params['latent_dim'] = 2
hyp_params['hidden_activation'] = tf.keras.activations.relu
hyp_params['bias_initializer'] = tf.keras.initializers.Zeros

# Encoding/Decoding Layer Parameters
hyp_params['num_en_layers'] = 2
hyp_params['num_en_neurons'] = 32
hyp_params['kernel_init_enc'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['kernel_init_dec'] = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
hyp_params['ae_output_activation'] = tf.keras.activations.linear

# Loss Function Parameters
hyp_params['a1'] = tf.constant(1e-1, dtype=hyp_params['precision'])  # Reconstruction
hyp_params['a2'] = tf.constant(1, dtype=hyp_params['precision'])  # Prediction
hyp_params['a3'] = tf.constant(1, dtype=hyp_params['precision'])  # Linearity
hyp_params['a4'] = tf.constant(1e-9, dtype=hyp_params['precision'])  # L-inf
hyp_params['a5'] = tf.constant(1e-14, dtype=hyp_params['precision'])  # L-2 on weights

# Learning rate
hyp_params['lr'] = 1e-3  # Learning rate

# Initialize the Koopman model and loss
myMachine = dl.DLDMD(hyp_params)
myLoss = lf.LossDLDMD(hyp_params)


# ==============================================================================
# Generate / load data
# ==============================================================================
data_fname = 'kdv_data.pkl'
if os.path.exists(data_fname):
    # Load data from file
    data = pickle.load(open(data_fname, 'rb'))
    data = tf.cast(data, dtype=hyp_params['precision'])
else:
    # Create new data
    data = dat.data_maker_kdv(x_lower1=-2, x_upper1=2, x_lower2=-2, x_upper2=2,
                              n_ic=hyp_params['num_init_conds'], dt=hyp_params['delta_t'],
                              tf=hyp_params['time_final'])
    data = tf.cast(data, dtype=hyp_params['precision'])
    # Save data to file
    pickle.dump(data, open(data_fname, 'wb'))

# Create training and validation datasets from the initial conditions
shuffled_data = tf.random.shuffle(data)
train_data = tf.data.Dataset.from_tensor_slices(shuffled_data[:hyp_params['num_train_init_conds'], :, :])
val_data = tf.data.Dataset.from_tensor_slices(shuffled_data[-hyp_params['batch_size']:, :, :])

# Batch and prefetch the validation data to the GPUs
val_set = val_data.batch(hyp_params['batch_size'], drop_remainder=True)
val_set = val_set.prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# Train the model
# ==============================================================================
results = tr.train_model(hyp_params=hyp_params, train_data=train_data,
                         val_set=val_set, model=myMachine, loss=myLoss)

print(results['model'].summary())
exit()
