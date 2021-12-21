import matplotlib.pyplot as plt


def plot_IVIM_param(D_star, D, Fp, s0_mask=1):
    """
    plot a figure of IVIM parameters map
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))

    D_star_plot = ax[0].imshow(D_star * s0_mask, cmap='gray', clim=(0, 0.1))
    ax[0].set_title('D*')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.colorbar(D_star_plot, ax=ax[0], fraction=0.046, pad=0.04)

    D_plot = ax[1].imshow(D * s0_mask, cmap='gray', clim=(0, 0.004))  #
    ax[1].set_title('D')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.colorbar(D_plot, ax=ax[1], fraction=0.046, pad=0.04)

    Fp_plot = ax[2].imshow(Fp * s0_mask, cmap='gray', clim=(0, 0.5))  #
    ax[2].set_title('Fp')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    fig.colorbar(Fp_plot, ax=ax[2], fraction=0.046, pad=0.04)

    plt.subplots_adjust(hspace=-0.5)
    plt.show()
    plt.show()
