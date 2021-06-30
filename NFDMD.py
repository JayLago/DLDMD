"""
    Author:
        Jay Lago, SDSU, 2021
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.regularizers import L2


class NFDMD(keras.Model):
    def __init__(self, hyp_params, **kwargs):
        super(NFDMD, self).__init__(**kwargs)

        # DMD parameters
        self.num_time_steps = int(hyp_params['num_time_steps'])
        self.num_pred_steps = int(hyp_params['num_pred_steps'])
        self.time_final = hyp_params['time_final']
        self.delta_t = hyp_params['delta_t']
        self.dmd_threshold = -6

        # Network parameters
        self.input_dim = hyp_params['input_dim']
        self.hidden_dim = hyp_params['hidden_dim']
        self.num_coupling_layers = hyp_params['num_coupling_layers']
        self.precision = hyp_params['precision']
        self.masks = np.array([[0, 1], [1, 0]] * (self.num_coupling_layers // 2), dtype=self.precision)
        self.coupling_layer_list = [self.coupling(self.input_dim) for ii in range(self.num_coupling_layers)]

    def coupling(self, input_shape):
        input = tf.keras.layers.Input(shape=(self.num_time_steps, self.input_dim))
        # reg = 0.01
        # # Shifting layers
        # shift_layer_1 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(input)
        # shift_layer_2 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(shift_layer_1)
        # shift_layer_3 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(shift_layer_2)
        # shift_layer_4 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(shift_layer_3)
        # shift_layer_5 = Dense(units=input_shape, activation="linear", kernel_regularizer=L2(reg))(shift_layer_4)
        # # Scaling layers
        # scale_layer_1 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(input)
        # scale_layer_2 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(scale_layer_1)
        # scale_layer_3 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(scale_layer_2)
        # scale_layer_4 = Dense(units=self.hidden_dim, activation="relu", kernel_regularizer=L2(reg))(scale_layer_3)
        # scale_layer_5 = Dense(units=input_shape, activation="tanh", kernel_regularizer=L2(reg))(scale_layer_4)

        shift_layer_1 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               input_shape=(self.num_time_steps, self.input_dim),
                               activation="relu")(input)
        shift_layer_2 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               activation="relu")(shift_layer_1)
        shift_layer_3 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               activation="relu")(shift_layer_2)
        shift_layer_4 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               activation="relu")(shift_layer_3)
        shift_layer_5 = Conv1D(filters=input_shape, kernel_size=(1,),
                               activation="relu")(shift_layer_4)
        scale_layer_1 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               input_shape=(self.num_time_steps, self.input_dim),
                               activation="relu")(input)
        scale_layer_2 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               activation="relu")(scale_layer_1)
        scale_layer_3 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               activation="relu")(scale_layer_2)
        scale_layer_4 = Conv1D(filters=self.hidden_dim, kernel_size=(1,),
                               activation="relu")(scale_layer_3)
        scale_layer_5 = Conv1D(filters=input_shape, kernel_size=(1,),
                               activation="relu")(scale_layer_4)

        return keras.Model(inputs=input, outputs=[scale_layer_5, shift_layer_5])

    def call(self, x):

        y = x

        for ii in range(self.num_coupling_layers)[::1]:
            x_masked = x * self.masks[ii]
            reversed_mask = 1 - self.masks[ii]
            s, t = self.coupling_layer_list[ii](x_masked)
            y = (x * tf.exp(s) + t)

        # Reshape for DMD step
        yt = tf.transpose(y, [0, 2, 1])

        # Generate latent time series using DMD prediction
        Phi, Lam, b, y_adv = self.dmd(yt)
        y_adv = tf.transpose(y_adv, [0, 2, 1])

        # Separate Re/Im parts
        y_adv_real = tf.math.real(y_adv)
        y_adv_imag = tf.math.imag(y_adv)

        # Decode the latent trajectories
        x_adv = self.shift_fn(y_adv_real)

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
                'scale_fn': self.scale_fn,
                'shift_fn': self.shift_fn}
