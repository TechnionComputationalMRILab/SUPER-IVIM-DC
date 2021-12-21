import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from SUPER_IVIM_DC.utils import isnan
from SUPER_IVIM_DC.utils.checkarg import checkarg
from SUPER_IVIM_DC.deep import Net


def infer_clinical_supervised_IVIM(X_infer, bvalues, ivim_path, arg, verbose=1):
    arg = checkarg(arg)

    # The b-values that get into the model need to be torch Tensor type.
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
    # load the pretrained network
    ivim_model = Net(bvalues, arg.net_pars).to(arg.train_pars.device)
    ivim_model.load_state_dict(torch.load(ivim_path))
    ivim_model.eval()

    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        selsb = np.array(bvalues) == 100
        S0 = np.nanmean(X_infer[:, selsb], axis=1)
        S0[S0 != S0] = 0
        S0 = np.squeeze(S0)
        valid_id = (S0 > (
                    0.5 * np.median(S0[S0 > 0])))  # Boolean parameter with indication for True/False prediction value.
        datatot = X_infer[valid_id, :]
    else:
        selsb = np.array(bvalues) == 0
        S0 = np.nanmean(X_infer[:, selsb], axis=1)
        S0[S0 != S0] = 0
        S0 = np.squeeze(S0)
        valid_id = (S0 > (
                    0.5 * np.median(S0[S0 > 0])))  # Boolean parameter with indication for True/False prediction value.
        datatot = X_infer[valid_id, :]

    # mylist = isnan(np.mean(X_infer, axis=1))
    # sels = [not i for i in mylist]
    # print(f'sels size is {len(sels)}')

    # normalise data
    S0 = np.nanmean(datatot[:, selsb], axis=1).astype('<f')
    datatot = datatot / S0[:, None]
    print(f'Clinical patient data loaded \n' if verbose > 0 else "", end='')

    # Limiting the percentile threshold

    # initialise parameters and data
    Dp_infer = np.array([])
    Dt_infer = np.array([])
    Fp_infer = np.array([])
    S0_infer = np.array([])
    recon_error = np.array([])

    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = DataLoader(torch.from_numpy(datatot.astype(np.float32)),
                                   batch_size=1,
                                   shuffle=False,
                                   drop_last=False)

    # start predicting
    with torch.no_grad():
        if verbose > 0:
            pbar = enumerate(tqdm(inferloader, position=0, leave=True), 0)
        else:
            pbar = enumerate(inferloader, 0)

        for i, (_, clinical_batch) in enumerate(pbar, 0):
            clinical_batch = clinical_batch.to(arg.train_pars.device)

            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the
            # predicted signal decay.
            clinical_infer, Dtt, Fpt, Dpt, S0t = ivim_model(clinical_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)

            Dp_infer = np.append(Dp_infer, (Dpt.cpu()).numpy())
            Dt_infer = np.append(Dt_infer, (Dtt.cpu()).numpy())
            Fp_infer = np.append(Fp_infer, (Fpt.cpu()).numpy())
            S0_infer = np.append(S0_infer, (S0t.cpu()).numpy())
            # this is an absulote error
            SR_error = np.sqrt(np.square(clinical_infer - clinical_batch)) / clinical_batch
            recon_error = np.append(recon_error, (SR_error.cpu()).numpy())
            # Error in precent -> devide the absolute error by the original value

    # switch between Dt & Dp if the prediction is wrong
    if np.mean(Dp_infer) < np.mean(Dt_infer):
        Dp22 = deepcopy(Dt_infer)
        Dt_infer = Dp_infer
        Dp_infer = Dp22
        Fp_infer = 1 - Fp_infer

    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    # here we correct for the data that initially was removed as it did not have IVIM behaviour, by returning zero
    # estimates
    Dp_out = np.zeros(len(valid_id))
    Dt_out = np.zeros(len(valid_id))
    Fp_out = np.zeros(len(valid_id))
    S0_out = np.zeros(len(valid_id))
    Dp_out[valid_id] = Dp_infer
    Dt_out[valid_id] = Dt_infer
    Fp_out[valid_id] = Fp_infer
    S0_out[valid_id] = S0_infer

    return [recon_error, Dp_out, Dt_out, Fp_out, S0_out]
