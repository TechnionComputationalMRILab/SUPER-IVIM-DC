import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from IVIMNET.deep import Net
from IVIMNET.utils import isnan
from IVIMNET.utils.checkarg import checkarg


def infer_supervised_IVIM(X_infer, labels, bvalues, ivim_path, arg, verbose=1):
    arg = checkarg(arg)
    n_bval = len(bvalues)
    n_samples = len(X_infer)
    print(f'The number of samples are: {n_samples} \n' if verbose > 0 else "", end='')

    # The b-values that get into the model need to be torch Tensor type.
    bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)

    # load the pretrained network
    ivim_model = Net(bvalues, arg.net_pars).to(arg.train_pars.device)
    ivim_model.load_state_dict(torch.load(ivim_path))
    ivim_model.eval()

    # normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_infer[:, bvalues == 100], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx = isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx, axis=0)
        labels = np.delete(labels, nan_idx, axis=0)  # Dt, f, Dp
        print(f'Phantom Training \n' if verbose > 0 else "", end='')
    else:
        S0 = np.mean(X_infer[:, bvalues == 0], axis=1).astype('<f')
        X_infer = X_infer / S0[:, None]
        nan_idx = isnan(np.mean(X_infer, axis=1))
        X_infer = np.delete(X_infer, nan_idx, axis=0)
        labels = np.delete(labels, nan_idx, axis=0)  # Dt, f, Dp

    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_300_idx = np.percentile(X_infer[:, bvalues < 500], 95, axis=1) < 1.3
        b_greater_300_idx = np.percentile(X_infer[:, bvalues > 500], 95, axis=1) < 1.2
        b_greater_700_idx = np.percentile(X_infer[:, bvalues > 1000], 95, axis=1) < 1
        thresh_idx = b_less_300_idx & b_greater_300_idx & b_greater_700_idx

    else:
        b_less_50_idx = np.percentile(X_infer[:, bvalues < 50], 95, axis=1) < 1.3
        b_greater_50_idx = np.percentile(X_infer[:, bvalues > 50], 95, axis=1) < 1.2
        b_greater_150_idx = np.percentile(X_infer[:, bvalues > 150], 95, axis=1) < 1
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx

    # combine the labels and the X_train data for supervised learning
    supervised_data = np.append(X_infer[thresh_idx,], labels[thresh_idx,], axis=1)

    # initialise parameters and data
    Dp_infer = np.array([])
    Dt_infer = np.array([])
    Fp_infer = np.array([])
    S0_infer = np.array([])
    Dp_orig = np.array([])
    Dt_orig = np.array([])
    Fp_orig = np.array([])
    S0_orig = np.array([1])

    # initialise dataloader. Batch size can be way larger as we are still training.
    inferloader = DataLoader(torch.from_numpy(supervised_data.astype(np.float32)),
                             batch_size=1,  # previously was 2056,
                             shuffle=False,
                             drop_last=False)

    # start predicting
    with torch.no_grad():
        if verbose > 0:
            pbar = enumerate(tqdm(inferloader, position=0, leave=True), 0)
        else:
            pbar = enumerate(inferloader, 0)

        for i, supervised_batch in pbar:

            supervised_batch = supervised_batch.to(arg.train_pars.device)
            X_batch = supervised_batch[:, :n_bval]

            Dp_batch = supervised_batch[:, -1]
            Fp_batch = supervised_batch[:, -2]
            Dt_batch = supervised_batch[:, -3]

            Dp_orig = np.append(Dp_orig, (Dp_batch.cpu()).numpy())
            Dt_orig = np.append(Dt_orig, (Dt_batch.cpu()).numpy())
            Fp_orig = np.append(Fp_orig, (Fp_batch.cpu()).numpy())

            # here the signal is predicted. Note that we now are interested in the parameters
            # and no longer in the predicted signal decay.
            _, Dtt, Fpt, Dpt, S0t = ivim_model(X_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except Exception as err:
                print(f'Error in infer_supervised_IVIM: {err} \n' if verbose > 0 else "", end='')
                S0 = np.append(S0, S0t)

            Dp_infer = np.append(Dp_infer, (Dpt.cpu()).numpy())
            Dt_infer = np.append(Dt_infer, (Dtt.cpu()).numpy())
            Fp_infer = np.append(Fp_infer, (Fpt.cpu()).numpy())
            S0_infer = np.append(S0_infer, (S0t.cpu()).numpy())
            # Error in precent -> divide the absolute error by the original value

    # switch between Dt & Dp if the prediction is wrong
    if np.mean(Dp_infer) < np.mean(Dt_infer):
        Dp22 = deepcopy(Dt_infer)
        Dt_infer = Dp_infer
        Dp_infer = Dp22
        Fp_infer = 1 - Fp_infer

    e_calc_type = 'NRSE'
    if e_calc_type == 'NRSE':
        Dp_norm_error = np.sqrt(np.square(Dp_orig - Dp_infer)) / Dp_orig
        Dt_norm_error = np.sqrt(np.square(Dt_orig - Dt_infer)) / Dt_orig
        Fp_norm_error = np.sqrt(np.square(Fp_orig - Fp_infer)) / Fp_orig
        S0_norm_error = np.sqrt(np.square(S0_orig - S0_infer))

    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    return [Dp_norm_error, Dt_norm_error, Fp_norm_error, S0_norm_error]
