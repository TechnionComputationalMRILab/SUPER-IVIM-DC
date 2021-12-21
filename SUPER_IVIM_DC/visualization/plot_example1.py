import numpy as np
import matplotlib.pyplot as plt


def plot_example1(paramsNN, paramsf, Dt_truth, Fp_truth, Dp_truth, arg, SNR):
    """
    taken from simulations.py
    """
    # initialise figure
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    if arg.fit.do_fit:
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    else:
        fig, ax = plt.subplots(2, 3, figsize=(20, 20))
    # fill Figure with values
    Dt_t_plot = ax[0, 0].imshow(Dt_truth, cmap='gray', clim=(0, 0.003))
    ax[0, 0].set_title('Dt, ground truth')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    fig.colorbar(Dt_t_plot, ax=ax[0, 0], fraction=0.046, pad=0.04)

    Dt_plot = ax[1, 0].imshow(np.reshape(paramsNN[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
    ax[1, 0].set_title('Dt, estimate')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(Dt_plot, ax=ax[1, 0], fraction=0.046, pad=0.04)

    if arg.fit.do_fit:
        Dt_fit_plot = ax[2, 0].imshow(np.reshape(paramsf[0], (sx, sy)), cmap='gray', clim=(0, 0.003))
        ax[2, 0].set_title('Dt, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 0].set_xticks([])
        ax[2, 0].set_yticks([])
        fig.colorbar(Dt_fit_plot, ax=ax[2, 0], fraction=0.046, pad=0.04)

    Fp_t_plot = ax[0, 1].imshow(Fp_truth, cmap='gray', clim=(0, 0.5))
    ax[0, 1].set_title('Fp, ground truth')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    fig.colorbar(Fp_t_plot, ax=ax[0, 1], fraction=0.046, pad=0.04)

    Fp_plot = ax[1, 1].imshow(np.reshape(paramsNN[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
    ax[1, 1].set_title('Fp, estimate')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(Fp_plot, ax=ax[1, 1], fraction=0.046, pad=0.04)

    if arg.fit.do_fit:
        Fp_fit_plot = ax[2, 1].imshow(np.reshape(paramsf[1], (sx, sy)), cmap='gray', clim=(0, 0.5))
        ax[2, 1].set_title('f, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 1].set_xticks([])
        ax[2, 1].set_yticks([])
        fig.colorbar(Fp_fit_plot, ax=ax[2, 1], fraction=0.046, pad=0.04)

    Dp_t_plot = ax[0, 2].imshow(Dp_truth, cmap='gray', clim=(0.01, 0.1))
    ax[0, 2].set_title('Dp, ground truth')
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    fig.colorbar(Dp_t_plot, ax=ax[0, 2], fraction=0.046, pad=0.04)

    Dp_plot = ax[1, 2].imshow(np.reshape(paramsNN[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
    ax[1, 2].set_title('Dp, estimate')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    fig.colorbar(Dp_plot, ax=ax[1, 2], fraction=0.046, pad=0.04)

    if arg.fit.do_fit:
        Dp_fit_plot = ax[2, 2].imshow(np.reshape(paramsf[2], (sx, sy)), cmap='gray', clim=(0.01, 0.1))
        ax[2, 2].set_title('Dp, fit {fitmethod}'.format(fitmethod=arg.fit.method))
        ax[2, 2].set_xticks([])
        ax[2, 2].set_yticks([])
        fig.colorbar(Dp_fit_plot, ax=ax[2, 2], fraction=0.046, pad=0.04)

        plt.subplots_adjust(hspace=0.2)
        plt.show()
    plt.savefig('plots/plot_imshow_IVIM_param_{snr}.png'.format(snr=SNR, save=arg.save_name))