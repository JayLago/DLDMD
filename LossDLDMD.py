"""
    Created by:
        Jay Lago
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MSE


class LossDLDMD(keras.losses.Loss):
    def __init__(self, hyp_params, **kwargs):
        super(LossDLDMD, self).__init__(**kwargs)

        # Parameters
        self.a1 = hyp_params['a1']
        self.a2 = hyp_params['a2']
        self.a3 = hyp_params['a3']
        self.a4 = hyp_params['a4']
        self.a5 = hyp_params['a5']
        self.global_batch_size = hyp_params['batch_size']
        self.precision = hyp_params['precision']
        self.pretrain = hyp_params['pretrain']

        # Loss components
        self.loss_recon = tf.constant(0.0, dtype=self.precision)
        self.loss_pred = tf.constant(0.0, dtype=self.precision)
        self.loss_dmd = tf.constant(0.0, dtype=self.precision)
        self.loss_inf = tf.constant(0.0, dtype=self.precision)
        self.loss_reg = tf.constant(0.0, dtype=self.precision)
        self.total_loss = tf.constant(0.0, dtype=self.precision)

    def call(self, model, obs):
        """
            model = [y, x_ae, x_adv, y_adv_real, y_adv_imag, weights, Lam, Phi, b]
        """
        if self.pretrain:
            # Autoencoder reconstruction
            x_ae = tf.identity(model[1])
            self.loss_recon = tf.reduce_mean(MSE(obs, x_ae))
            self.total_loss = self.a1*self.loss_recon
        else:
            y = tf.identity(model[0])
            x_ae = tf.identity(model[1])
            x_adv = tf.identity(model[2])
            y_adv = tf.identity(model[3])
            weights = model[5]
            pred_horizon = -1

            # Autoencoder reconstruction
            self.loss_recon = tf.reduce_mean(MSE(obs, x_ae))

            # Future state prediction
            self.loss_pred = tf.reduce_mean(MSE(obs[:, :pred_horizon, :], x_adv[:, :pred_horizon, :]))

            # DMD reconstruction in the latent space
            self.loss_dmd = tf.reduce_mean(MSE(y[:, :pred_horizon, :], y_adv[:, :pred_horizon, :]))

            # L-inf penalty
            self.loss_inf = tf.reduce_max(tf.abs(obs[:, 0, :] - x_ae[:, 0, :])) + \
                            tf.reduce_max(tf.abs(obs[:, 1, :] - x_adv[:, 1, :]))

            # Regularization on weights
            self.loss_reg = tf.add_n([tf.nn.l2_loss(w) for w in weights])

            # Total loss
            self.total_loss = self.a1*self.loss_recon + self.a2*self.loss_pred + \
                              self.a3*self.loss_dmd + self.a4*self.loss_inf + \
                              self.a5*self.loss_reg

        return self.total_loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'loss_recon': self.loss_recon,
                'loss_pred': self.loss_pred,
                'loss_dmd': self.loss_dmd,
                'loss_inf': self.loss_inf,
                'loss_reg': self.loss_reg,
                'total_loss': self.total_loss}


class LossDLEDMD(keras.losses.Loss):
    def __init__(self, hyp_params, **kwargs):
        super(LossDLEDMD, self).__init__(**kwargs)

        # Parameters
        self.a1 = hyp_params['a1']
        self.a2 = hyp_params['a2']
        self.a3 = hyp_params['a3']
        self.a4 = hyp_params['a4']
        self.a5 = hyp_params['a5']
        self.global_batch_size = hyp_params['batch_size']
        self.precision = hyp_params['precision']
        self.pretrain = hyp_params['pretrain']

        # Loss components
        self.loss_recon = tf.constant(0.0, dtype=self.precision)
        self.loss_pred = tf.constant(0.0, dtype=self.precision)
        self.loss_dmd = tf.constant(0.0, dtype=self.precision)
        self.loss_inf = tf.constant(0.0, dtype=self.precision)
        self.loss_reg = tf.constant(0.0, dtype=self.precision)
        self.total_loss = tf.constant(0.0, dtype=self.precision)

    def call(self, model, obs):
        """
            model = [y, x_ae, x_adv, y_adv_real, y_adv_imag, weights, Lam, Phi, b]
        """
        if self.pretrain:
            # Autoencoder reconstruction
            x_ae = tf.identity(model[1])
            self.loss_recon = tf.reduce_mean(MSE(obs, x_ae))
            self.total_loss = self.a1*self.loss_recon
        else:
            y = tf.identity(model[0])
            x_ae = tf.identity(model[1])
            x_adv = tf.identity(model[2])
            y_adv = tf.identity(model[3])
            weights = model[5]
            pred_horizon = -1

            # Autoencoder reconstruction
            self.loss_recon = tf.reduce_mean(MSE(obs, x_ae))

            # Future state prediction
            # self.loss_pred = tf.reduce_mean(MSE(obs[:, :pred_horizon, :], x_adv[:, :pred_horizon, :]))

            # EDMD reconstruction in the latent space
            self.loss_dmd = tf.reduce_mean(MSE(y[:, :pred_horizon, :], y_adv[:, :pred_horizon, :]))

            # L-inf penalty
            self.loss_inf = tf.reduce_max(tf.abs(obs[:, 0, :] - x_ae[:, 0, :])) + \
                            tf.reduce_max(tf.abs(obs[:, 1, :] - x_adv[:, 1, :]))

            # Regularization on weights
            self.loss_reg = tf.add_n([tf.nn.l2_loss(w) for w in weights])

            # Total loss
            self.total_loss = self.a1*self.loss_recon + self.a3*self.loss_dmd + \
                              self.a4*self.loss_inf + self.a5*self.loss_reg

        return self.total_loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'loss_recon': self.loss_recon,
                'loss_pred': self.loss_pred,
                'loss_dmd': self.loss_dmd,
                'loss_inf': self.loss_inf,
                'loss_reg': self.loss_reg,
                'total_loss': self.total_loss}
