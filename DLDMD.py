"""
    Author:
        Jay Lago, SDSU, 2021
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


class DLDMD(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(DLDMD, self).__init__(**kwargs)

        # Parameters
        self.batch_size = hyp_params['batch_size']
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
        self.precision = hyp_params['precision']
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
            x_adv = tf.zeros(shape=x.shape, dtype=self.precision)
            y_adv_real = tf.zeros(shape=y.shape, dtype=self.precision)
            y_adv_imag = tf.zeros(shape=y.shape, dtype=self.precision)
            Lam = tf.zeros(shape=(self.batch_size, self.latent_dim), dtype=tf.complex128)
            Phi = tf.zeros(shape=(self.batch_size, self.latent_dim, self.latent_dim), dtype=tf.complex128)
            b = tf.zeros(shape=(self.batch_size, self.latent_dim), dtype=tf.complex128)
        else:
            # Reshape for DMD step
            yt = tf.transpose(y, [0, 2, 1])

            # Generate latent time series using DMD prediction
            Phi, Lam, b, y_adv = self.dmd(yt)
            y_adv = tf.transpose(y_adv, [0, 2, 1])

            # Separate Re/Im parts
            y_adv_real = tf.math.real(y_adv)
            y_adv_imag = tf.math.imag(y_adv)

            # Decode the latent trajectories
            x_adv = self.decoder(y_adv_real)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv_real, y_adv_imag, weights, Lam, Phi, b]

    @tf.function
    def dmd(self, y):
        y_m = y[:, :, :-1]
        y_p = y[:, :, 1:]
        S, U, V = tf.linalg.svd(y_m, compute_uv=True, full_matrices=False)
        # smax = tf.reduce_max(S)
        # idx = self.log10(S/smax) > self.dmd_threshold
        S = tf.linalg.diag(S)
        r = S.shape[-1]
        Si = tf.linalg.pinv(S)
        U = U[:, :, :r]
        V = V[:, :, :r]
        Uh = tf.transpose(U, conjugate=True, perm=[0, 2, 1])
        A = Uh @ (y_p @ (V @ Si))
        Lam, W = tf.linalg.eig(A)
        Phi = tf.cast(((y_p @ V) @ Si), dtype=tf.complex128) @ W
        Phi_inv = self.cpinv(Phi)
        Omega = tf.math.log(Lam) / self.delta_t
        y0 = tf.cast(y_m[:, :, 0], dtype=tf.complex128)
        b = tf.linalg.matvec(Phi_inv, y0)
        Psi = tf.TensorArray(tf.complex128, size=self.num_pred_steps)
        tpred = tf.cast(tf.linspace(0.0, self.time_final, self.num_pred_steps), dtype=tf.complex128)
        ii = 0
        for tstep in tpred:
            Psi = Psi.write(ii, tf.math.multiply(tf.math.exp(Omega * tstep), b))
            ii += 1
        pred = Phi @ tf.transpose(Psi.stack(), perm=[1, 2, 0])
        return Phi, Lam, b, pred

    @tf.function
    def cpinv(self, X):
        S, U, V = tf.linalg.svd(X, compute_uv=True)
        Si = tf.cast(tf.linalg.diag(1 / S), dtype=tf.complex128)
        return V @ Si @ tf.transpose(U, conjugate=True, perm=[0, 2, 1])

    @tf.function
    def log10(self, x):
        return tf.math.log(x) / tf.math.log(tf.constant(10., dtype=x.dtype))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder}
