from IVIMNET.utils.vtk_read import vtk_read
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage


def clinical_signal(slice_number, case_1_path):
    """
    This simulates IVIM curves. Data is simulated by randomly selecting a value of D, f and D* from within the
    predefined range.

    :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)

    """

    dw_mri_files = []
    for dw_file in os.listdir(case_1_path):
        if os.path.isfile(os.path.join(case_1_path, dw_file)):
            if dw_file.endswith('.vtk'):  # take only vtk extention
                dw_mri_files.append(os.path.join(case_1_path, dw_file))
                # print(dw_mri_files)

    b_val = np.array([0, 50, 100, 200, 400, 600, 800])
    clinic_signal = vtk_read(dw_mri_files, b_val, slice_number, 0, 0, "case 004 plot")
    sb, sx, sy = clinic_signal.shape

    s0_image = clinic_signal[0, :, :]
    backgournd = s0_image[0:30, 0:30]
    signal_SNR = backgournd.mean() / backgournd.std()
    print(f'Background_SNR {signal_SNR}')
    plt.hist(backgournd, bins=np.arange(50))
    image_SNR = s0_image.mean() / s0_image.std()
    print(f'image_SNR {image_SNR}')
    plt.hist(s0_image, bins=np.arange(50))
    s0_mask = np.ones_like(s0_image)
    s0_mask[s0_image < 20] = 0  # case 004,009 is s0_image<15 - case 005,006,007 is s0_image<20
    s0_mask = ndimage.median_filter(s0_mask, size=20)
    # plt.imshow(s0_mask, cmap = 'gray')
    np.expand_dims(s0_mask, axis=-1)

    out_signal = clinic_signal.transpose().reshape(-1, sb)  # list of pixels according to the image columns

    return out_signal, sb, sx, sy, b_val, s0_mask, clinic_signal