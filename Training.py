"""
    Created by:
        Jay Lago - 23 Dec 2020
"""
import os
import pickle
import time
import datetime as dt
import numpy as np
import tensorflow as tf

import HelperFuns as hf


def train_model(hyp_params, train_data, val_set, model, loss):
    # Dictionary to store all relevant training parameters and losses
    train_params = dict()
    train_params['start_time'] = time.time()
    train_params['train_loss_results'] = []
    train_params['val_loss_results'] = []
    train_params['val_loss_comps_avgs'] = []

    for epoch in range(1, hyp_params['max_epochs'] + 1):
        epoch_start_time = dt.datetime.now()
        epoch_time = time.time()
        epoch_loss_avg_train = tf.keras.metrics.Mean()
        epoch_loss_avg_val = tf.keras.metrics.Mean()

        # Shuffle, batch, and prefetch training data to the GPU
        train_set = train_data.shuffle(hyp_params['num_train_init_conds']) \
            .batch(hyp_params['batch_size'], drop_remainder=True)
        train_set = train_set.prefetch(tf.data.AUTOTUNE)

        # Set optimizer
        if hyp_params['optimizer'] == 'adam':
            myoptimizer = tf.keras.optimizers.Adam(hyp_params['lr'])
        if hyp_params['optimizer'] == 'sgd':
            myoptimizer = tf.keras.optimizers.SGD(learning_rate=hyp_params['lr'], momentum=0.9)

        # Pretraining the autoencoder
        if hyp_params['pretrain'] and epoch < hyp_params['num_pretrain']:
            print("pretraining...")
            loss.a1 = tf.constant(1.0, dtype=hyp_params['precision'])  # AE
            loss.a2 = tf.constant(0.0, dtype=hyp_params['precision'])  # X predictions
            loss.a3 = tf.constant(0.0, dtype=hyp_params['precision'])  # Y predictions
            loss.a4 = tf.constant(0.0, dtype=hyp_params['precision'])  # max norm on x_adv/x_ae
            loss.a5 = tf.constant(1e-6, dtype=hyp_params['precision'])  # W regularization
        else:
            model.pretrain = False
            loss.pretrain = False
            loss.a1 = hyp_params['a1']  # AE
            loss.a2 = hyp_params['a2']  # X predictions
            loss.a3 = hyp_params['a3']  # Y predictions
            loss.a4 = hyp_params['a4']  # max norm on x_adv/x_ae
            loss.a5 = hyp_params['a5']  # W regularization

        # Begin batch training
        with tf.device(hyp_params['device']):
            for train_batch in train_set:
                with tf.GradientTape() as tape:
                    train_pred = model(train_batch, training=True)
                    train_loss = loss(train_pred, train_batch)
                gradients = tape.gradient(train_loss, model.trainable_weights)
                myoptimizer.apply_gradients([(grad, var) for (grad, var) in zip(gradients, model.trainable_weights)
                                             if grad is not None])
                myoptimizer.apply_gradients(zip(gradients, model.trainable_weights))
                epoch_loss_avg_train.update_state(train_loss)

            # Batch validation
            lrecon = tf.keras.metrics.Mean()
            lpred = tf.keras.metrics.Mean()
            ldmd = tf.keras.metrics.Mean()
            linf = tf.keras.metrics.Mean()
            for val_batch in val_set:
                val_pred = model(val_batch)
                val_loss = loss(val_pred, val_batch)
                epoch_loss_avg_val.update_state(val_loss)
                # Save loss components for diagnostic plotting
                lrecon.update_state(np.log10(loss.loss_recon))
                lpred.update_state(np.log10(loss.loss_pred))
                ldmd.update_state(np.log10(loss.loss_dmd))
                linf.update_state(np.log10(loss.loss_inf))
            train_params['val_loss_comps_avgs'].append([lrecon.result(), lpred.result(),
                                                        ldmd.result(), linf.result()])

        # Report training status
        train_params['train_loss_results'].append(np.log10(epoch_loss_avg_train.result()))
        train_params['val_loss_results'].append(np.log10(epoch_loss_avg_val.result()))
        print("Epoch {epoch} of {max_epoch} / Train {train:3.7f} / Val {test:3.7f} / LR {lr:2.7f} / {time:4.2f} seconds"
              .format(epoch=epoch, max_epoch=hyp_params['max_epochs'],
                      train=train_params['train_loss_results'][-1],
                      test=train_params['val_loss_results'][-1],
                      lr=hyp_params['lr'],
                      time=time.time() - epoch_time))

        # Save training diagnostic plots
        if not model.pretrain:
            if epoch == 1 or epoch % hyp_params['plot_every'] == 0:
                if not os.path.exists(hyp_params['plot_path']):
                    os.makedirs(hyp_params['plot_path'])
                this_plot = hyp_params['plot_path'] + '/' + epoch_start_time.strftime("%Y%m%d%H%M%S") + '.png'
                hf.diagnostic_plot(val_pred, val_batch, hyp_params, epoch,
                                   this_plot, train_params['val_loss_comps_avgs'],
                                   train_params['val_loss_results'])

        # Save model
        if epoch % hyp_params['save_every'] == 0 or epoch == hyp_params['max_epochs']:
            if not os.path.exists(hyp_params['model_path']):
                os.makedirs(hyp_params['model_path'])
            model_path = hyp_params['model_path'] + '/epoch_{epoch}_loss_{loss:2.3}' \
                .format(epoch=epoch, loss=train_params['val_loss_results'][-1])
            model.save_weights(model_path + '.h5')
            pickle.dump(hyp_params, open(model_path + '.pkl', 'wb'))

    print("\nTotal training time: %4.2f minutes" % ((time.time() - train_params['start_time']) / 60.0))
    print("Final train loss: %2.7f" % (train_params['train_loss_results'][-1]))
    print("Final validation loss: %2.7f" % (train_params['val_loss_results'][-1]))

    results = dict()
    results['model'] = model
    results['loss'] = loss
    results['val_loss_history'] = train_params['val_loss_results']
    results['val_loss_comps'] = train_params['val_loss_comps_avgs']

    return results
