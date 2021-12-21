import numpy as np


def load_phantom_data(dirpath, phantom_data_filename="PhantomData_forExp2.npz"):
    data = np.load(f'{dirpath}\datasets\{phantom_data_filename}')

    b_100_im = data['b_100_im']
    b_300_im = data['b_300_im']
    b_500_im = data['b_500_im']
    b_700_im = data['b_700_im']
    b_900_im = data['b_900_im']
    b_1100_im = data['b_1100_im']
    b_1300_im = data['b_1300_im']
    b_1500_im = data['b_1500_im']
    b_1700_im = data['b_1700_im']
    b_1900_im = data['b_1900_im']
    b_2100_im = data['b_2100_im']
    b_2300_im = data['b_2300_im']
    b_2500_im = data['b_2500_im']

    dwmri_stack = np.stack((b_100_im, b_300_im, b_500_im, b_700_im,
                            b_900_im, b_1100_im, b_1300_im, b_1500_im,
                            b_1700_im, b_1900_im, b_2100_im, b_2300_im, b_2500_im))
    dwmri_stack = dwmri_stack.astype(np.int64)

    return dwmri_stack
