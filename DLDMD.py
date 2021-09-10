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
        if self.precision == 'float32':
            self.precision_complex = tf.complex64
        else:
            self.precision_complex = tf.complex128
        self.dmd_threshold = -10

        # Construct the ENCODER network
        self.encoder = keras.Sequential(name="encoder")
        self.encoder.add(Dense(self.num_neurons,
                               input_shape=self.enc_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_in'))
        for ii in range(self.num_en_layers):
            self.encoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_enc'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='enc_' + str(ii)))
        self.encoder.add(Dense(self.latent_dim,
                               activation=hyp_params['ae_output_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='enc_out'))

        # Construct the DECODER network
        self.decoder = keras.Sequential(name="decoder")
        self.decoder.add(Dense(self.num_neurons,
                               input_shape=self.dec_input,
                               activation=hyp_params['hidden_activation'],
                               kernel_initializer=hyp_params['kernel_init_enc'],
                               bias_initializer=hyp_params['bias_initializer'],
                               trainable=True, name='dec_in'))
        for ii in range(self.num_en_layers):
            self.decoder.add(Dense(self.num_neurons,
                                   activation=hyp_params['hidden_activation'],
                                   kernel_initializer=hyp_params['kernel_init_dec'],
                                   bias_initializer=hyp_params['bias_initializer'],
                                   trainable=True, name='dec_' + str(ii)))
        self.decoder.add(Dense(self.phys_dim,
                               activation=hyp_params['ae_output_activation'],
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
            Lam = tf.zeros(shape=(self.batch_size, self.latent_dim), dtype=self.precision_complex)
            Phi = tf.zeros(shape=(self.batch_size, self.latent_dim, self.latent_dim), dtype=self.precision_complex)
            b = tf.zeros(shape=(self.batch_size, self.latent_dim), dtype=self.precision_complex)
        else:
            # Reshape for DMD step
            yt = tf.transpose(y, [0, 2, 1])

            # Generate latent time series using DMD prediction
            Phi, Lam, b, y_adv = self.edmd(yt)

            # Separate Re/Im parts
            y_adv_real = tf.math.real(y_adv)
            y_adv_imag = tf.math.imag(y_adv)

            # Decode the latent trajectories
            x_adv = self.decoder(y_adv_real)

        # Model weights
        weights = self.trainable_weights

        return [y, x_ae, x_adv, y_adv_real, y_adv_imag, weights, Lam, Phi, b]

    def edmd(self, Y):
        Y_m = Y[:, :, :-1]
        Y_p = Y[:, :, 1:]

        sig, U, V = tf.linalg.svd(Y_m, compute_uv=True, full_matrices=False)
        r = sig.shape[-1]  # Normally use threshold, but we're not using DMD for reduction
        sigr_inv = tf.linalg.diag(1.0 / sig[:, :r])
        Ur = U[:, :, :r]
        Urh = tf.linalg.adjoint(Ur)
        Vr = V[:, :, :r]

        A = Y_p @ Vr @ sigr_inv @ Urh
        evals, evecs = tf.linalg.eig(A)
        Phi = tf.linalg.solve(evecs, tf.cast(Y_m, dtype=self.precision_complex))
        y0 = Phi[:, :, 0]
        y0 = y0[:, :, tf.newaxis]

        recon = tf.TensorArray(self.precision_complex, size=self.num_pred_steps)
        recon = recon.write(0, evecs @ y0)
        evals_k = tf.identity(evals)
        for ii in tf.range(1, self.num_pred_steps):
            tmp = evecs @ (tf.linalg.diag(evals_k) @ y0)
            recon = recon.write(ii, tmp)
            evals_k = evals_k * evals
        recon = tf.transpose(tf.squeeze(recon.stack()), perm=[1, 0, 2])
        return Phi, evals, evecs, recon

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'encoder': self.encoder,
                'decoder': self.decoder}
