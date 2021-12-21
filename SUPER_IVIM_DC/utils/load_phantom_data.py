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

    # background
    bkg_ROI = dwmri_stack[12, 0:10, 0:10]
    bkg_SNR = bkg_ROI.mean() / bkg_ROI.std()

    ROI4 = dwmri_stack[:, 48:54, 30:36]
    ROI5 = dwmri_stack[:, 48:54, 60:66]
    ROI6 = dwmri_stack[:, 51:57, 93:99]
    ROI7 = dwmri_stack[:, 78:84, 30:36]
    ROI8 = dwmri_stack[:, 80:86, 60:66]
    ROI9 = dwmri_stack[:, 81:87, 93:99]
    ROI10 = dwmri_stack[:,114:120 , 25:31]
    ROI11 = dwmri_stack[:,111:117 ,61:67 ]
    ROI12 = dwmri_stack[:,111:117 ,92:98 ]
    names = ["Dairy cream 38%", "Non-Dairy cream 21%", "Egg yolk",
             "Water", "Dairy cream 9%", "Egg white & yolk",
             "Acetone", "Dairy cream 15%", "3 Egg white & 1 yolk"]

    batch_ROI = [ROI4, ROI5, ROI6, ROI7, ROI8, ROI9, ROI10, ROI11, ROI12]

    return dict(zip(names, batch_ROI)), bkg_SNR
