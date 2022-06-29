import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot import plot_IVIM_signal
from utiles import create_working_folder
from IVIMNET.deep import Net
import IVIMNET.deep as deep
import IVIMNET.simulations as sim
from hyperparams import hyperparams as hp

def train_model(key, arg, mode, sf, work_dir):

    SNR = arg.sim.SNR[0]

    if (mode == 'SUPER-IVIM-DC'):
        supervised = True
        if (key == 'fetal'):
            arg.loss_coef_ivim = 0.4
        else:
            coef = [0.1,0.1,0.2,0.35,0.4] #[0.09, 0.1, 0.2, 0.18, 0.25, 0.4]
            arg.loss_coef_ivim = coef[sf-1]
        arg.train_pars.ivim_combine = True
        arg = deep.checkarg(arg)

    elif (mode == 'IVIMNET'):
        supervised = False
        arg.train_pars.ivim_combine = False
        arg = deep.checkarg(arg)

    matNN,_  = sim.sim(SNR, arg, supervised, sf, mode, work_dir)

    return matNN
