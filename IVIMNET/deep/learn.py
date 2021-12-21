import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import copy
import os
import matplotlib.pyplot as plt

from IVIMNET.utils.isnan import isnan
from IVIMNET.utils.checkarg import checkarg
from IVIMNET.deep import Net
from IVIMNET.deep import load_optimizer
from IVIMNET.visualization.plot_progress import plot_progress
from IVIMNET.visualization.loss_plot_supervised import loss_plot_supervised


def learn_IVIM(X_train, bvalues, arg, net=None, save_state_model=False, save_as=""):
    """
    This program builds a IVIM-NET network and trains it.
    :param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second
        axis are the b-values
    :param bvalues: a 1D array with the b-values
    :param arg: an object with network design options, as explained in the publication check hyperparameters.py for
    options
    :param net: an optional input pre-trained network with initialized weights for e.g. transfer
        learning or warm start
    :param save_state_model: if the model should be saved
    :param save_as: path where to save the model
    :return net: returns a trained network
    """

    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    ## normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_train[:, bvalues == 100], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)
        print('phantom training')
    else:
        S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        np.delete(X_train, isnan(np.mean(X_train, axis=1)), axis=0)

    # removing non-IVIM-like data; this often gets through when background data is not correctly masked
    # Estimating IVIM parameters in these data is meaningless anyways.
    if arg.key == 'phantom':
        X_train = X_train[np.percentile(X_train[:, bvalues < 500], 95, axis=1) < 1.3]
        X_train = X_train[np.percentile(X_train[:, bvalues > 500], 95, axis=1) < 1.2]
        X_train = X_train[np.percentile(X_train[:, bvalues > 1000], 95, axis=1) < 1.0]

    else:
        X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3]
        X_train = X_train[np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2]
        X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
    X_train[X_train > 1.5] = 1.5

    # initialising the network of choice using the input argument arg
    if net is None:
        bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
        net = Net(bvalues, arg.net_pars).to(arg.train_pars.device)
    else:
        # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
        net.to(arg.train_pars.device)

    # defining the loss function; not explored in the publication
    if arg.train_pars.loss_fun == 'rms':
        criterion = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    elif arg.train_pars.loss_fun == 'L1':
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                       [split, len(X_train) - split])
    # train loader loads the trianing data. We want to shuffle to make sure data order is modified each
    # epoch and different data is selected each epoch.
    trainloader = DataLoader(train_set,
                             batch_size=arg.train_pars.batch_size,
                             shuffle=True,
                             drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation
    # every time. We can use substantially more data per batch as we are not training.
    inferloader = DataLoader(val_set,
                             batch_size=32 * arg.train_pars.batch_size,
                             shuffle=False,
                             drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    prev_lr = 0
    # get_ipython().run_line_magic('matplotlib', 'inline')
    final_model = copy.deepcopy(net.state_dict())

    ## Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        # initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_val = 0.
        # losstotcon = 0.
        maxloss = 0.
        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            X_batch = X_batch.to(arg.train_pars.device)
            ## forward + backward + optimize
            X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred = net(X_batch)
            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # determine loss for batch; note that the loss is determined by the difference between the predicted
            # signal and the actual signal. The loss does not look at Dt, Dp or Fp.
            loss = criterion(X_pred, X_batch)
            # updating network
            loss.backward()
            optimizer.step()
            # total loss and determine max loss over all batches
            running_loss_train += loss.item()
            if loss.item() > maxloss:
                maxloss = loss.item()
        # show some figures if desired, to show whether there is a correlation between Dp and f
        if arg.fig:
            plt.figure(3)
            plt.clf()
            plt.plot(Dp_pred.tolist(), Fp_pred.tolist(), 'rx', markersize=5)
            plt.ion()
            plt.show()
        # after training, do validation in unseen data without updating gradients
        print('\n validation \n')
        net.eval()
        # validation is always done over all validation data
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            X_batch = X_batch.to(arg.train_pars.device)
            # do prediction, only look at predicted IVIM signal
            X_pred, _, _, _, _ = net(X_batch)
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # validation loss
            loss = criterion(X_pred, X_batch)
            running_loss_val += loss.item()
        # scale losses
        running_loss_train = running_loss_train / totalit
        running_loss_val = running_loss_val / batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        # as discussed in the article, LR is important. This approach allows to reduce the LR if we think it is too
        # high, and return to the network state before it went poorly
        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']
        # print stuff
        print("\nLoss: {loss}, validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                             val_loss=running_loss_val,
                                                                             lr=optimizer.param_groups[0]['lr']))
        # early stopping criteria
        if running_loss_val < best:
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.train_pars.patience:
                print("\nDone, best val loss: {}".format(best))
                break

    print("Done")

    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/fig_fit.png')
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/fig_train.png')
        plt.close('all')

    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    if save_state_model:
        final_weights = net.state_dict()
        torch.save(final_weights, save_as)
    return net


def learn_supervised_IVIM(X_train, labels, bvalues, arg, net=None, save_state_model=False, save_as=""):
    """
    This program builds supervised IVIM-NET network and trains it.
    :param labels:
    :param bvalues: a 1D array with the b-values
    :param arg: an object with network design options, as explained in the publication
        check hyperparameters.py for options
    :param net: an optional input pre-trained network with initialized weights for
        e.g. transfer learning or warm start
    :param save_state_model: if the model should be saved
    :param save_as: path where to save the model
    :return net: returns a trained network
    """
    # ivim_combine = False
    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)
    n_bval = len(bvalues)
    ivim_combine = arg.train_pars.ivim_combine
    print(f'\n \n ivim combine flag is {ivim_combine}\n \n ')

    # normalise the signal to b=0 and remove data with nans
    if arg.key == 'phantom':
        S0 = np.mean(X_train[:, bvalues == 100], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        nan_idx = isnan(np.mean(X_train, axis=1))
        X_train = np.delete(X_train, nan_idx, axis=0)
        labels = np.delete(labels, nan_idx, axis=0)  # Dt, f, Dp
        print('phantom training')
    else:
        S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype('<f')
        X_train = X_train / S0[:, None]
        nan_idx = isnan(np.mean(X_train, axis=1))
        X_train = np.delete(X_train, nan_idx, axis=0)
        labels = np.delete(labels, nan_idx, axis=0)  # Dt, f, Dp

    # Limiting the percentile threshold
    if arg.key == 'phantom':
        b_less_300_idx = np.percentile(X_train[:, bvalues < 500], 95, axis=1) < 1.3
        b_greater_300_idx = np.percentile(X_train[:, bvalues > 500], 95, axis=1) < 1.2
        b_greater_700_idx = np.percentile(X_train[:, bvalues > 1000], 95, axis=1) < 1
        thresh_idx = b_less_300_idx & b_greater_300_idx & b_greater_700_idx

    else:
        b_less_50_idx = np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.3
        b_greater_50_idx = np.percentile(X_train[:, bvalues > 50], 95, axis=1) < 1.2
        b_greater_150_idx = np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1
        thresh_idx = b_less_50_idx & b_greater_50_idx & b_greater_150_idx

    suprevised_data = np.append(X_train[thresh_idx,], labels[thresh_idx,],
                                axis=1)  # combine the labels and the X_train data for supervised learning

    # initialising the network of choice using the input argument arg
    if net is None:
        bvalues = torch.FloatTensor(bvalues[:]).to(arg.train_pars.device)
        net = Net(bvalues, arg.net_pars, supervised=True).to(arg.train_pars.device)
    else:
        # if a network was used as input parameter, work with that network instead (transfer learning/warm start).
        net.to(arg.train_pars.device)

    # defining the loss function; not explored in the publication
    criterion_Dt = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    criterion_Fp = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    criterion_Dp = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    if ivim_combine:
        criterion_ivim = nn.MSELoss(reduction='mean').to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(suprevised_data.astype(np.float32)),
                                                       [split, len(suprevised_data) - split])

    # train loader loads the trianing data. We want to shuffle to make sure data order is modified
    # each epoch and different data is selected each epoch.
    trainloader = DataLoader(train_set,
                             batch_size=arg.train_pars.batch_size,
                             shuffle=True,
                             drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for
    # validation every time. We can use substantially more data per batch as we are not training.
    inferloader = DataLoader(val_set,
                             batch_size=32 * arg.train_pars.batch_size,
                             shuffle=False,
                             drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    acum_loss_Dp = []
    acum_loss_Dt = []
    acum_loss_Fp = []
    val_loss_Dp = []
    val_loss_Dt = []
    val_loss_Fp = []
    if ivim_combine:
        acum_loss_recon = []
        val_loss_recon = []
    prev_lr = 0

    final_model = copy.deepcopy(net.state_dict())

    ## Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        # initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_Dp = 0.
        running_loss_Dt = 0.
        running_loss_Fp = 0.

        running_loss_val = 0.
        running_loss_Dp_val = 0.
        running_loss_Dt_val = 0.
        running_loss_Fp_val = 0.

        if ivim_combine:
            running_loss_recon = 0.
            running_loss_recon_val = 0.
        # losstotcon = 0.
        maxloss = 0.
        for i, suprevised_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            # forward + backward + optimize
            X_batch = suprevised_batch[:, :n_bval]

            Dp_batch = suprevised_batch[:, -1]
            Fp_batch = suprevised_batch[:, -2]
            Dt_batch = suprevised_batch[:, -3]
            # farward path
            X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred = net(X_batch)

            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3

            # determine loss for batch; note that the loss is determined by the difference between the
            # predicted signal and the actual signal. The loss does not look at Dt, Dp or Fp.
            Dp_loss = criterion_Dp(Dp_pred, Dp_batch.unsqueeze(1))
            Fp_loss = criterion_Fp(Fp_pred, Fp_batch.unsqueeze(1))
            Dt_loss = criterion_Dt(Dt_pred, Dt_batch.unsqueeze(1))
            # S0_loss = criterion_Dt(S0_pred, S0_batch)
            if ivim_combine:
                ivim_loss = criterion_ivim(X_pred, X_batch)

            # loss coefficiants
            Dp_coeff = arg.loss_coef_Dp
            Dt_coeff = arg.loss_coef_Dt
            Fp_coeff = arg.loss_coef_Fp
            if ivim_combine:
                ivim_coeff = arg.loss_coef_ivim
                loss = Dp_coeff * Dp_loss + Dt_coeff * Dt_loss + Fp_coeff * Fp_loss + ivim_coeff * ivim_loss
            else:
                loss = Dp_coeff * Dp_loss + Dt_coeff * Dt_loss + Fp_coeff * Fp_loss

            # updating network
            loss.backward()
            optimizer.step()
            # parameters loss
            running_loss_Dp += Dp_loss.item()
            running_loss_Dt += Dt_loss.item()
            running_loss_Fp += Fp_loss.item()
            if ivim_combine:
                running_loss_recon += ivim_loss.item()
            # total loss and determine max loss over all batches
            running_loss_train += loss.item()
            if loss.item() > maxloss:
                maxloss = loss.item()
        # show some figures if desired, to show whether there is a correlation between Dp and f
        if arg.fig:
            plt.figure()
            plt.clf()
            plt.plot(Dp_pred.tolist(), Fp_pred.tolist(), 'rx', markersize=5)
            plt.ion()
            plt.show()
        # after training, do validation in unseen data without updating gradients
        print('\n validation \n')
        net.eval()
        # validation is always done over all validation data
        for i, suprevised_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            suprevised_batch = suprevised_batch.to(arg.train_pars.device)
            X_batch = suprevised_batch[:, :n_bval]  # TODO verify that the data in X_btcs cet print(first line)
            ## forward + backward + optimize
            Dp_batch = suprevised_batch[:, -1]
            Fp_batch = suprevised_batch[:, -2]
            Dt_batch = suprevised_batch[:, -3]

            X_pred, Dt_pred, Fp_pred, Dp_pred, S0pred = net(X_batch)

            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # validation loss
            Dp_loss_val = criterion_Dp(Dp_pred, Dp_batch.unsqueeze(1))
            Fp_loss_val = criterion_Fp(Fp_pred, Fp_batch.unsqueeze(1))
            Dt_loss_val = criterion_Dt(Dt_pred, Dt_batch.unsqueeze(1))
            if ivim_combine:
                ivim_loss_val = criterion_ivim(X_pred, X_batch)
                loss = Dp_coeff * Dp_loss_val + Dt_coeff * Dt_loss_val + Fp_coeff * Fp_loss_val + \
                       ivim_coeff * ivim_loss_val
            else:
                loss = Dp_coeff * Dp_loss_val + Dt_coeff * Dt_loss_val + Fp_coeff * Fp_loss_val  # add the wieghts
                # as configurable value
            running_loss_val += loss.item()
            running_loss_Dp_val += Dp_loss_val.item()
            running_loss_Dt_val += Dt_loss_val.item()
            running_loss_Fp_val += Fp_loss_val.item()
            if ivim_combine:
                running_loss_recon_val += ivim_loss_val.item()
        # scale losses
        running_loss_train = running_loss_train / totalit
        running_loss_Dp = running_loss_Dp / totalit
        running_loss_Dt = running_loss_Dt / totalit
        running_loss_Fp = running_loss_Fp / totalit
        running_loss_val = running_loss_val / batch_norm2
        running_loss_Dp_val = running_loss_Dp / batch_norm2
        running_loss_Dt_val = running_loss_Dt / batch_norm2
        running_loss_Fp_val = running_loss_Fp / batch_norm2

        if ivim_combine:
            running_loss_recon += running_loss_recon / totalit
            running_loss_recon_val += running_loss_recon_val / batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        acum_loss_Dp.append(running_loss_Dp)
        acum_loss_Dt.append(running_loss_Dt)
        acum_loss_Fp.append(running_loss_Fp)
        val_loss_Dp.append(running_loss_Dp_val)
        val_loss_Dt.append(running_loss_Dt_val)
        val_loss_Fp.append(running_loss_Fp_val)

        if ivim_combine:
            acum_loss_recon.append(running_loss_recon)
            val_loss_recon.append(running_loss_recon_val)
            loss_plot_supervised(acum_loss_Dp, acum_loss_Dt, acum_loss_Fp, loss_train, loss_val,
                                 val_loss_Dp, val_loss_Dt, val_loss_Fp, acum_loss_recon, val_loss_recon)
        else:
            loss_plot_supervised(acum_loss_Dp, acum_loss_Dt, acum_loss_Fp, loss_train, loss_val,
                                 val_loss_Dp, val_loss_Dt, val_loss_Fp)
        # as discussed in the article, LR is important. This approach allows to reduce the LR if we think it is too
        # high, and return to the network state before it went poorly
        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']
        # print stuff
        print("\n IVIM Loss: {loss}, IVIM validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                                        val_loss=running_loss_val,
                                                                                        lr=optimizer.param_groups[0][
                                                                                            'lr']))
        # print(f'\n D* Loss: {running_loss_Dp}')
        # print(f'\n D Loss: {running_loss_Dt}')
        # print(f'\n Fp Loss: {running_loss_Fp}')
        # early stopping criteria
        if running_loss_val < best:  # TODO change it to total loss
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.train_pars.patience:
                print("\nDone, best val loss: {}".format(best))
                break
        # plot loss and plot 4 fitted curves
        if epoch > 0:
            # plot progress and intermediate results (if enabled)
            plot_progress(X_batch.cpu(), X_pred.cpu(), bvalues.cpu(), loss_train, loss_val, arg)

    print("Done")
    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/fig_fit.png')
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/fig_train.png')
        plt.close('all')
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    # save the model
    if save_state_model:
        final_weights = net.state_dict()
        torch.save(final_weights, save_as)

    return net
