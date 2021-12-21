import torch.optim as optim
import torch


def load_optimizer(net, arg):
    if arg.net_pars.parallel:
        if arg.net_pars.fitS0:
            par_list = [{'params': net.encoder.parameters(), 'lr': arg.train_pars.lr},
                        {'params': net.encoder2.parameters()}, {'params': net.encoder3.parameters()},
                        {'params': net.encoder4.parameters()}]
        else:
            par_list = [{'params': net.encoder.parameters(), 'lr': arg.train_pars.lr},
                        {'params': net.encoder2.parameters()}, {'params': net.encoder3.parameters()}]
    else:
        par_list = [{'params': net.encoder.parameters()}]

    if arg.train_pars.optim == 'adam':
        optimizer = optim.Adam(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    elif arg.train_pars.optim == 'sgd':
        optimizer = optim.SGD(par_list, lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
    elif arg.train_pars.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)

    if arg.train_pars.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
                                                         patience=round(arg.train_pars.patience / 2))
        return optimizer, scheduler
    else:
        return optimizer
