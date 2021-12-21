import numpy as np
import matplotlib.pyplot as plt
import os
from IVIMNET.utils.ivim_functions import ivim


def sim_signal_predict(arg, SNR):
    # init randomstate
    rg = np.random.RandomState(123)
    # define parameter values in the three regions
    S0_region0, S0_region1, S0_region2 = 1, 1, 1
    Dp_region0, Dp_region1, Dp_region2 = 0.03, 0.05, 0.07
    Dt_region0, Dt_region1, Dt_region2 = 0.0020, 0.0015, 0.0010
    Fp_region0, Fp_region1, Fp_region2 = 0.15, 0.3, 0.45
    # image size
    sx, sy, sb = 100, 100, len(arg.sim.bvalues)
    # create image
    dwi_image = np.zeros((sx, sy, sb))
    Dp_truth = np.zeros((sx, sy))
    Dt_truth = np.zeros((sx, sy))
    Fp_truth = np.zeros((sx, sy))

    # fill image with simulated values
    for i in range(sx):
        for j in range(sy):
            if (40 < i < 60) and (40 < j < 60):
                # region 0
                dwi_image[i, j, :] = ivim(arg.sim.bvalues, Dt_region0, Fp_region0, Dp_region0, S0_region0)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region0, Dt_region0, Fp_region0
            elif (20 < i < 80) and (20 < j < 80):
                # region 1
                dwi_image[i, j, :] = ivim(arg.sim.bvalues, Dt_region1, Fp_region1, Dp_region1, S0_region1)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region1, Dt_region1, Fp_region1
            else:
                # region 2
                dwi_image[i, j, :] = ivim(arg.sim.bvalues, Dt_region2, Fp_region2, Dp_region2, S0_region2)
                Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region2, Dt_region2, Fp_region2

    # plot simulated diffusion weighted image
    fig, ax = plt.subplots(2, int(np.round(arg.sim.bvalues.shape[0] / 2)), figsize=(20, 20))
    b_id = 0
    for i in range(2):
        for j in range(int(np.round(arg.sim.bvalues.shape[0] / 2))):
            if not b_id == arg.sim.bvalues.shape[0]:
                ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('b = ' + str(arg.sim.bvalues[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            else:
                # ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('End of b-values')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            b_id += 1
    plt.subplots_adjust(hspace=0)
    plt.show()
    if not os.path.isdir('plots'):
        os.makedirs('plots')
    plt.savefig('plots/plot_dwi_without_noise_param_{snr}_{method}.png'.format(snr=SNR, method=arg.save_name))

    # Initialise dwi noise image
    dwi_noise_imag = np.zeros((sx, sy, sb))
    # fill dwi noise image with Gaussian noise
    for i in range(sx):
        for j in range(sy):
            dwi_noise_imag[i, j, :] = rg.normal(0, 1 / SNR, (1, len(arg.sim.bvalues)))
    # Add Gaussian noise to dwi image
    dwi_image_noise = dwi_image + dwi_noise_imag
    # normalise signal
    S0_dwi_noisy = np.mean(dwi_image_noise[:, :, arg.sim.bvalues == 0], axis=1)
    dwi_image_noise_norm = dwi_image_noise / S0_dwi_noisy[:, None]

    # plot simulated diffusion weighted image with noise
    fig, ax = plt.subplots(2, int(np.round(arg.sim.bvalues.shape[0] / 2)), figsize=(20, 20))
    b_id = 0
    for i in range(2):
        for j in range(int(np.round(arg.sim.bvalues.shape[0] / 2))):
            if not b_id == arg.sim.bvalues.shape[0]:
                ax[i, j].imshow(dwi_image_noise_norm[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('b = ' + str(arg.sim.bvalues[b_id]))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            else:
                # ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1))
                ax[i, j].set_title('End of b-values')
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
            b_id += 1
    plt.subplots_adjust(hspace=0)
    plt.show()
    plt.savefig('plots/plot_dwi_with_noise_param_{snr}_{method}.png'.format(snr=SNR, method=arg.save_name))

    # reshape image
    dwi_image_long = np.reshape(dwi_image_noise_norm, (sx * sy, sb))
    return dwi_image_long, Dt_truth, Fp_truth, Dp_truth
