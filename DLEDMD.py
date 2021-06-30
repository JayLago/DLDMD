"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

class DLEDMD(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(DLEDMD, self).__init__(**kwargs)

        # Parameters
        self.phys_dim = hyp_params['phys_dim']
        self.latent_dim = hyp_params['latent_dim']
        self.num_time_steps = int(hyp_params['num_time_steps'])
        self.num_pred_steps = int(hyp_params['num_pred_steps'])
        self.time_final = hyp_params['time_final']
        self.num_en_layers = hyp_params['num_en_layers']
        self.num_neurons = hyp_params['num_en_neurons']
        self.delta_t = hyp_params['delta_t']
        self.kernel_size = 1
        self.enc_input = (self.num_time_steps, self.phys_dim)
        self.dec_input = (self.num_time_steps, self.latent_dim)
        self.pretrain = hyp_params['pretrain']
        self.dmd_threshold = -6

        # Construct the ENCODER network
        self.encoder = keras.Sequential(name="encoder")
        self.encoder.add(Conv1D(filters=self.num_neurons,
                                kernel_size=(self.kernel_size,),
                                input_shape=self.enc_input,
                                activation=hyp_params['hidden_activation'],
                                kernel_initializer=hyp_params['kernel_init_enc'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='enc_in'))
        for ii in range(self.num_en_layers - 1):
            self.encoder.add(Conv1D(self.num_neurons,
                                    kernel_size=(self.kernel_size,),
                                    activation=hyp_params['hidden_activation'],
                                    padding='same',
                                    kernel_initializer=hyp_params['kernel_init_enc'],
                                    bias_initializer=hyp_params['bias_initializer'],
                                    trainable=True, name='enc_' + str(ii)))
        self.encoder.add(Conv1D(self.latent_dim,
                                kernel_size=(self.kernel_size,),
                                activation=hyp_params['ae_output_activation'],
                                padding='same',
                                kernel_initializer=hyp_params['kernel_init_enc'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='enc_out'))

        # Construct the DECODER network
        self.decoder = keras.Sequential(name="decoder")
        self.decoder.add(Conv1D(filters=self.num_neurons,
                                kernel_size=(self.kernel_size,),
                                input_shape=self.dec_input,
                                activation=hyp_params['hidden_activation'],
                                padding='same',
                                kernel_initializer=hyp_params['kernel_init_enc'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='dec_in'))
        for ii in range(self.num_en_layers - 1):
            self.decoder.add(Conv1D(self.num_neurons,
                                    kernel_size=(self.kernel_size,),
                                    activation=hyp_params['hidden_activation'],
                                    padding='same',
                                    kernel_initializer=hyp_params['kernel_init_dec'],
                                    bias_initializer=hyp_params['bias_initializer'],
                                    trainable=True, name='dec_' + str(ii)))
        self.decoder.add(Conv1D(self.phys_dim,
                                kernel_size=(self.kernel_size,),
                                activation=hyp_params['ae_output_activation'],
                                padding='same',
                                kernel_initializer=hyp_params['kernel_init_dec'],
                                bias_initializer=hyp_params['bias_initializer'],
                                trainable=True, name='dec_out'))

    def call(self, x):
        # Encoder on the entire time series
        y = self.encoder(x)
        x_ae = self.decoder(y)

        if self.pretrain:
            x_adv, y_adv_real, y_adv_imag = None, None, None
            k_evals, k_efuns, k_modes = None, None, None

        else:
            # Transpose for DMD step
            yt = tf.transpose(y, [0, 2, 1])
            xt = tf.transpose(x, [0, 2, 1])

            # Generate latent time series using DMD prediction
            k_efuns, k_evals, k_modes, y_adv = self.edmd(yt, xt)

            # Separate Re/Im parts
            y_adv_real = tf.math.real(y_adv)
            y_adv_imag = tf.math.imag(y_adv)

            # Decode the latent trajectories
            x_adv = self.decoder(y_adv_real)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv_real, y_adv_imag, weights, k_evals, k_efuns, k_modes]

    # @tf.function
    def edmd(self, y, x):
        y_m = y[:, :, :-1]
        y_p = y[:, :, 1:]

        # SVD step (w/ optional rank truncation)
        S, U, V = tf.linalg.svd(y_m, full_matrices=True, compute_uv=True)
        # smax = tf.reduce_max(S)
        # r = self.log10(S/smax) > self.dmd_threshold
        r = S.shape[-1]
        Si = tf.linalg.diag(1.0/S)
        U = U[:, :, :r]
        V = V[:, :, :r]
        Uh = tf.linalg.adjoint(U)

        # Koopman eigenvalues, eigenfunctions, and modes
        K = y_p @ (V @ (Si @ Uh))
        k_evals, k_modes = tf.linalg.eig(K)
        k_evals = tf.math.log(k_evals) / self.delta_t
        psi = tf.cast(x, dtype=tf.complex128)
        k_efuns = tf.linalg.solve(k_modes, psi)

        # Prediction step
        xint = k_efuns[:, :, -1]
        xint = xint[:, :, tf.newaxis]
        evals = tf.square(tf.linalg.diag(k_evals))
        y_pred = tf.TensorArray(tf.complex128, size=self.num_pred_steps)
        for tt in tf.range(self.num_pred_steps):
            y_pred = y_pred.write(tt, k_modes @ evals @ xint)
            evals = tf.math.multiply(evals, evals)
        y_pred = tf.transpose(tf.squeeze(y_pred.stack()), perm=[1, 0, 2])
        return k_efuns, k_evals, k_modes, y_pred

    @tf.function
    def log10(self, x):
        return tf.math.log(x) / tf.math.log(tf.constant(10., dtype=x.dtype))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder}




# import matplotlib.pyplot as plt
# import numpy as np
#
# ee = tf.reshape(k_evals, 256*2)
#
# plt.figure(1)
# plt.plot(np.real(ee), np.imag(ee), '.')
# plt.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), 'r--')
# plt.show()
