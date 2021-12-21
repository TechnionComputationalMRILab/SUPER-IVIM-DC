import numpy as np
import os

from IVIMNET.utils.hyperparams import hyperparams as hp_example
from IVIMNET.simulations.sim_signal import sim_signal
from IVIMNET.utils.checkarg import checkarg
from IVIMNET.visualization.boxplot_ivim import boxplot_ivim
from IVIMNET.inference.supervised_ivim import infer_supervised_IVIM
from pathlib import Path

DIR_PATH = os.path.dirname(os.path.realpath(os.path.abspath('')))

SNR = 10
arg = hp_example()
arg = checkarg(arg)
b_values = arg.sim.bvalues
sample_size = [10, 50, 100, 250, 500, 1000, 2000]

for n_samples in sample_size:
    IVIM_signal_noisy, D, f, Dp = sim_signal(SNR, b_values, n_samples, Dmin=arg.sim.range[0][0],
                                             Dmax=arg.sim.range[1][0], fmin=arg.sim.range[0][1],
                                             fmax=arg.sim.range[1][1], Dsmin=arg.sim.range[0][2],
                                             Dsmax=arg.sim.range[1][2], rician=arg.sim.rician)

    labels = np.stack((D, f, Dp), axis=1).squeeze()

    # select only relevant values, delete background and noise, and normalise data
    IVIMNET_path = Path(f'{DIR_PATH}/saved/models/IVIMNET_saved_models/SNR_10_IVIMNET.pt')
    IVIMSUPER_path = Path(f'{DIR_PATH}/saved/models/IVIMNET_saved_models/SNR10_IVIMSUPER.pt')

    DtNET_error, FpNET_error, DpNET_error, S0NET_error = infer_supervised_IVIM(IVIM_signal_noisy, labels, b_values,
                                                                               IVIMNET_path, arg)
    DtSUPER_error, FpSUPER_error, DpSUPER_error, S0SUPER_error = infer_supervised_IVIM(IVIM_signal_noisy, labels,
                                                                                       b_values, IVIMSUPER_path, arg)

    errors_np_array = np.stack([DpNET_error, DpSUPER_error, DtNET_error, DtSUPER_error,
                                FpNET_error, FpSUPER_error, ], axis=1)
    bp_title = "IVIMNET VS IVIMSUPER parameters error SNR=10"

    boxplot_ivim(errors_np_array, bp_title)
