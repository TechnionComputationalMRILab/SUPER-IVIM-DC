import numpy as np
import matplotlib.pyplot as plt


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
    """
    this program plots the progress of the training. It will plot the loss and validation
    loss, as well as 4 IVIM curve

    fits to 4 data points from the input
    """

    inds1 = np.argsort(bvalues)
    X_batch = X_batch[:, inds1]
    X_pred = X_pred[:, inds1]
    bvalues = bvalues[inds1]

    if arg.fig:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(bvalues, X_batch.data[0], 'o')
        axs[0, 0].plot(bvalues, X_pred.data[0])
        axs[0, 0].set_ylim(min(X_batch.data[0]) - 0.3, 1.2 * max(X_batch.data[0]))
        axs[1, 0].plot(bvalues, X_batch.data[1], 'o')
        axs[1, 0].plot(bvalues, X_pred.data[1])
        axs[1, 0].set_ylim(min(X_batch.data[1]) - 0.3, 1.2 * max(X_batch.data[1]))
        axs[0, 1].plot(bvalues, X_batch.data[2], 'o')
        axs[0, 1].plot(bvalues, X_pred.data[2])
        axs[0, 1].set_ylim(min(X_batch.data[2]) - 0.3, 1.2 * max(X_batch.data[2]))
        axs[1, 1].plot(bvalues, X_batch.data[3], 'o')
        axs[1, 1].plot(bvalues, X_pred.data[3])
        axs[1, 1].set_ylim(min(X_batch.data[3]) - 0.3, 1.2 * max(X_batch.data[3]))
        plt.legend(('data', 'estimate from network'))
        for ax in axs.flat:
            ax.set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
        for ax in axs.flat:
            ax.label_outer()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.figure(2)
        plt.clf()
        plt.plot(loss_train)
        plt.plot(loss_val)
        plt.yscale("log")
        plt.xlabel('epoch #')
        plt.ylabel('loss')
        plt.legend(('training loss', 'validation loss (after training epoch)'))
        plt.ion()
        plt.show()
        plt.pause(0.001)
