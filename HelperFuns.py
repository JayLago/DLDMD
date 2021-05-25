"""
    Created by:
        Jay Lago

    Modified:
        17 Nov 2020 - Jay Lago
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import numpy as np

font = {'family': 'DejaVu Sans', 'size': 18}
matplotlib.rc('font', **font)


def diagnostic_plot(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    if hyp_params['experiment'] == 'discrete' or \
            hyp_params['experiment'] == 'pendulum' or \
            hyp_params['experiment'] == 'van_der_pol':
        plot_2D(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss)
    elif hyp_params['experiment'] == 'fluid_flow_slow' or \
            hyp_params['experiment'] == 'fluid_flow_full':
        plot_3d(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss)
    else:
        print("unknown experiment, create new diagnostic plots")


def plot_2D(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(40, 20))
    ax = ax.flat
    skip = 1

    # Validation batch
    for ii in np.arange(0, y_true.shape[0], skip):
        ax[0].plot(y_true[ii, :, 0], y_true[ii, :, 1], '-')
    ax[0].scatter(y_true[:, 0, 0], y_true[:, 0, 1])
    ax[0].grid()
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ax[1].plot(enc_adv_dec[ii, :, 0], enc_adv_dec[ii, :, 1], '-')
    ax[1].scatter(enc_adv_dec[:, 0, 0], enc_adv_dec[:, 0, 1])
    ax[1].grid()
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")
    ax[1].set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    for ii in np.arange(0, enc.shape[0], skip):
        ax[2].plot(enc[ii, :, 0], enc[ii, :, 1], '-')
    ax[2].scatter(enc[:, 0, 0], enc[:, 0, 1])
    ax[2].grid()
    ax[2].set_xlabel("y1")
    ax[2].set_ylabel("y2")
    ax[2].axis("equal")
    ax[2].set_title("Encoded (y)")

    # Encoded-decoded time series
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ax[3].plot(enc_dec[ii, :, 0], enc_dec[ii, :, 1], '-')
    ax[3].scatter(enc_dec[:, 0, 0], enc_dec[:, 0, 1])
    ax[3].grid()
    ax[3].set_xlabel("x1")
    ax[3].set_ylabel("x2")
    ax[3].set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ax[4].plot(enc_adv[ii, :, 0], enc_adv[ii, :, 1], '-')
    ax[4].scatter(enc_adv[:, 0, 0], enc_adv[:, 0, 1])
    ax[4].grid()
    ax[4].set_xlabel("y1")
    ax[4].set_ylabel("y2")
    ax[4].axis("equal")
    ax[4].set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax[5].plot(val_loss, color='k', linewidth=lw, label='total')
    ax[5].set_title("Total Loss")
    ax[5].grid()
    ax[5].set_xlabel("Epoch")
    ax[5].set_ylabel("$log_{10}(L)$")
    ax[5].legend(loc="upper right")

    ax[6].plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax[6].set_title("Recon Loss")
    ax[6].grid()
    ax[6].set_xlabel("Epoch")
    ax[6].set_ylabel("$log_{10}(L_{recon})$")
    ax[6].legend(loc="upper right")

    ax[7].plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax[7].set_title("Prediction Loss")
    ax[7].grid()
    ax[7].set_xlabel("Epoch")
    ax[7].set_ylabel("$log_{10}(L_{pred})$")
    ax[7].legend(loc="upper right")

    ax[8].plot(loss_comps[:, 2], color='g', linewidth=lw, label='lin')
    ax[8].set_title("Linearity")
    ax[8].grid()
    ax[8].set_xlabel("Epoch")
    ax[8].set_ylabel("$log_{10}(L_{lin})$")
    ax[8].legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))

    plt.savefig(save_path)
    plt.close()


def plot_3d(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    font = {'family': 'DejaVu Sans', 'size': 10}
    matplotlib.rc('font', **font)

    skip = 8
    fig = plt.figure(figsize=(40, 20))

    # Validation batch
    ax = fig.add_subplot(3, 3, 1, projection='3d')
    for ii in np.arange(0, y_true.shape[0], skip):
        ii = int(ii)
        x1 = y_true[ii, :, 0]
        x2 = y_true[ii, :, 1]
        x3 = y_true[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    ax = fig.add_subplot(3, 3, 2, projection='3d')
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv_dec[ii, :, 0]
        x2 = enc_adv_dec[ii, :, 1]
        x3 = enc_adv_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    ax = fig.add_subplot(3, 3, 3, projection='3d')
    for ii in np.arange(0, enc.shape[0], skip):
        ii = int(ii)
        x1 = enc[ii, :, 0]
        x2 = enc[ii, :, 1]
        x3 = enc[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded (y)")

    # Encoded-decoded time series
    ax = fig.add_subplot(3, 3, 4, projection='3d')
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_dec[ii, :, 0]
        x2 = enc_dec[ii, :, 1]
        x3 = enc_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    ax = fig.add_subplot(3, 3, 5, projection='3d')
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv[ii, :, 0]
        x2 = enc_adv[ii, :, 1]
        x3 = enc_adv[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(val_loss, color='k', linewidth=lw, label='total')
    ax.set_title("Total Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L)$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax.set_title("Recon Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{recon})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax.set_title("Prediction Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{pred})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(loss_comps[:, 2], color='g', linewidth=lw, label='lin')
    ax.set_title("Linearity")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{lin})$")
    ax.legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))

    plt.savefig(save_path)
    plt.close()


def plot_fluid(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    font = {'family': 'DejaVu Sans', 'size': 10}
    matplotlib.rc('font', **font)

    skip = 10
    fig = plt.figure(figsize=(40, 20))

    # Validation batch
    ax = fig.add_subplot(3, 3, 1, projection='3d')
    for ii in np.arange(0, y_true.shape[0], skip):
        ii = int(ii)
        x1 = y_true[ii, :, 0]
        x2 = y_true[ii, :, 1]
        x3 = y_true[ii, :, 2]
        ax.plot3D(x1, x2, x3, 'gray')
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    ax = fig.add_subplot(3, 3, 2, projection='3d')
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv_dec[ii, :, 0]
        x2 = enc_adv_dec[ii, :, 1]
        x3 = enc_adv_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3, 'gray')
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    ax = fig.add_subplot(3, 3, 3)
    for ii in np.arange(0, enc.shape[0], skip):
        ii = int(ii)
        x1 = enc[ii, :, 0]
        x2 = enc[ii, :, 1]
        ax.plot(x1, x2)
    ax.axis('equal')
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_title("Encoded (y)")

    # Encoded-decoded time series
    ax = fig.add_subplot(3, 3, 4, projection='3d')
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ii = int(ii)
        x1 = enc_dec[ii, :, 0]
        x2 = enc_dec[ii, :, 1]
        x3 = enc_dec[ii, :, 2]
        ax.plot3D(x1, x2, x3, 'gray')
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_zlabel("$x_{3}$")
    ax.set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    ax = fig.add_subplot(3, 3, 5)
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv[ii, :, 0]
        x2 = enc_adv[ii, :, 1]
        ax.plot(x1, x2)
    ax.axis('equal')
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(val_loss, color='k', linewidth=lw, label='total')
    ax.set_title("Total Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L)$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax.set_title("Recon Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{recon})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax.set_title("Prediction Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{pred})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(loss_comps[:, 2], color='g', linewidth=lw, label='lin')
    ax.set_title("Linearity")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{lin})$")
    ax.legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))

    plt.savefig(save_path)
    plt.close()


def plot_vdp(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    font = {'family': 'DejaVu Sans', 'size': 10}
    matplotlib.rc('font', **font)

    skip = 8
    fig = plt.figure(figsize=(40, 20))

    # Validation batch
    ax = fig.add_subplot(3, 3, 1)
    for ii in np.arange(0, y_true.shape[0], skip):
        ax.plot(y_true[ii, :, 0], y_true[ii, :, 1])
    ax.axis('equal')
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    ax = fig.add_subplot(3, 3, 2)
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ax.plot(enc_adv_dec[ii, :, 0], enc_adv_dec[ii, :, 1])
    ax.axis('equal')
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    ax = fig.add_subplot(3, 3, 3, projection='3d')
    for ii in np.arange(0, enc.shape[0], skip):
        ii = int(ii)
        x1 = enc[ii, :, 0]
        x2 = enc[ii, :, 1]
        x3 = enc[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded (y)")

    # Encoded-decoded time series
    ax = fig.add_subplot(3, 3, 4)
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ax.plot(enc_dec[ii, :, 0], enc_dec[ii, :, 1])
    ax.set_xlabel("$x_{1}$")
    ax.set_ylabel("$x_{2}$")
    ax.axis('equal')
    ax.set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    ax = fig.add_subplot(3, 3, 5, projection='3d')
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ii = int(ii)
        x1 = enc_adv[ii, :, 0]
        x2 = enc_adv[ii, :, 1]
        x3 = enc_adv[ii, :, 2]
        ax.plot3D(x1, x2, x3)
    ax.set_xlabel("$y_{1}$")
    ax.set_ylabel("$y_{2}$")
    ax.set_zlabel("$y_{3}$")
    ax.set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(val_loss, color='k', linewidth=lw, label='total')
    ax.set_title("Total Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L)$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 7)
    ax.plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax.set_title("Recon Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{recon})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 8)
    ax.plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax.set_title("Prediction Loss")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{pred})$")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(3, 3, 9)
    ax.plot(loss_comps[:, 2], color='g', linewidth=lw, label='lin')
    ax.set_title("Linearity")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$log_{10}(L_{lin})$")
    ax.legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))

    plt.savefig(save_path)
    plt.close()


def plot_pendulum_uniform(y_pred, y_true, hyp_params, epoch, save_path, loss_comps, val_loss):
    enc = y_pred[0]
    enc_dec = y_pred[1]
    enc_adv_dec = y_pred[2]
    enc_adv = y_pred[3]

    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(40, 20))
    ax = ax.flat
    skip = 1

    # Validation batch
    for ii in np.arange(0, y_true.shape[0], skip):
        ax[0].plot(y_true[ii, :, 0], y_true[ii, :, 1], '-')
    ax[0].scatter(y_true[:, 0, 0], y_true[:, 0, 1])
    ax[0].grid()
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].set_title("Validation Data (x)")

    # Encoded-advanced-decoded time series
    for ii in np.arange(0, enc_adv_dec.shape[0], skip):
        ax[1].plot(enc_adv_dec[ii, :, 0], enc_adv_dec[ii, :, 1], '-')
    ax[1].scatter(enc_adv_dec[:, 0, 0], enc_adv_dec[:, 0, 1])
    ax[1].grid()
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")
    ax[1].set_title("Encoded-Advanced-Decoded (x_adv))")

    # Encoded time series
    for ii in np.arange(0, enc.shape[0], skip):
        ax[2].plot(enc[ii, :, 0], enc[ii, :, 1], '-')
    ax[2].scatter(enc[:, 0, 0], enc[:, 0, 1])
    ax[2].grid()
    ax[2].set_xlabel("y1")
    ax[2].set_ylabel("y2")
    ax[2].axis("equal")
    ax[2].set_title("Encoded (y)")

    # Encoded-decoded time series
    for ii in np.arange(0, enc_dec.shape[0], skip):
        ax[3].plot(enc_dec[ii, :, 0], enc_dec[ii, :, 1], '-')
    ax[3].scatter(enc_dec[:, 0, 0], enc_dec[:, 0, 1])
    ax[3].grid()
    ax[3].set_xlabel("x1")
    ax[3].set_ylabel("x2")
    ax[3].set_title("Encoded-Decoded (x_ae)")

    # Encoded-advanced time series
    for ii in np.arange(0, enc_adv.shape[0], skip):
        ax[4].plot(enc_adv[ii, :, 0], enc_adv[ii, :, 1], '-')
    ax[4].scatter(enc_adv[:, 0, 0], enc_adv[:, 0, 1])
    ax[4].grid()
    ax[4].set_xlabel("y1")
    ax[4].set_ylabel("y2")
    ax[4].axis("equal")
    ax[4].set_title("Encoded-Advanced (y_adv))")

    # Loss components
    lw = 3
    loss_comps = np.asarray(loss_comps)
    ax[5].plot(val_loss, color='k', linewidth=lw, label='total')
    ax[5].set_title("Total Loss")
    ax[5].grid()
    ax[5].set_xlabel("Epoch")
    ax[5].set_ylabel("$log_{10}(L)$")
    ax[5].legend(loc="upper right")

    ax[6].plot(loss_comps[:, 0], color='r', linewidth=lw, label='recon')
    ax[6].set_title("Recon Loss")
    ax[6].grid()
    ax[6].set_xlabel("Epoch")
    ax[6].set_ylabel("$log_{10}(L_{recon})$")
    ax[6].legend(loc="upper right")

    ax[7].plot(loss_comps[:, 1], color='b', linewidth=lw, label='pred')
    ax[7].set_title("Prediction Loss")
    ax[7].grid()
    ax[7].set_xlabel("Epoch")
    ax[7].set_ylabel("$log_{10}(L_{pred})$")
    ax[7].legend(loc="upper right")

    ax[8].plot(loss_comps[:, 2], color='g', linewidth=lw, label='lin')
    ax[8].set_title("Linearity")
    ax[8].grid()
    ax[8].set_xlabel("Epoch")
    ax[8].set_ylabel("$log_{10}(L_{lin})$")
    ax[8].legend(loc="upper right")

    fig.suptitle(
        "Epoch: {cur_epoch}/{max_epoch}, Learn Rate: {lr:.5f}, Val. Loss: {loss:.3f}".format(
            cur_epoch=epoch,
            max_epoch=hyp_params['max_epochs'],
            lr=hyp_params['lr'],
            loss=val_loss[-1]))

    plt.savefig(save_path)
    plt.close()
