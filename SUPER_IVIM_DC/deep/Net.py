import torch
import torch.nn as nn
import copy


# Define the neural network.
class Net(nn.Module):
    def __init__(self, bvalues, net_pars, supervised = False):
        """
        this defines the Net class which is the network we want to train.
        :param bvalues: a 1D array with the b-values
        :param net_pars: an object with network design options, as explained in the publication, with attributes:
        fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
        times len(bvalues), with data sorted per voxel. This option was not explored in the publication
        dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
        batch_norm --> Boolean determining whether to use batch normalisation
        parallel --> Boolean determining whether to use separate networks for estimating the different IVIM parameters
        (True), or have them all estimated by a single network (False)
        con --> string which determines what type of constraint is used for the parameters. Options are:
        'sigmoid' allowing a sigmoid constraint
        'abs' having the absolute of the estimated values to constrain parameters to be positive
        'none' giving no constraints
        cons_min --> 1D array, if sigmoid is the constraint, these values give [Dmin, fmin, D*min, S0min]
        cons_max --> 1D array, if sigmoid is the constraint, these values give [Dmax, fmax, D*max, S0max]
        depth --> integer giving the network depth (number of layers)
        """
        super(Net, self).__init__()
        self.supervised = supervised
        self.bvalues = bvalues
        self.net_pars = net_pars
        if self.net_pars.width == 0:
            self.net_pars.width = len(bvalues)
        # define number of parameters being estimated
        self.est_pars = 3
        if self.net_pars.fitS0:
            self.est_pars += 1
        # define number of outputs, if neighbours are taken along, we expect 9 outputs, otherwise 1
        self.outs = 1
        # define module lists. If network is not parallel, we can do with 1 list, otherwise we need a list per parameter
        self.fc_layers = nn.ModuleList()
        if self.net_pars.parallel:
            self.fc_layers2 = nn.ModuleList()
            self.fc_layers3 = nn.ModuleList()
            self.fc_layers4 = nn.ModuleList()
        # loop over the layers
        width = len(bvalues)
        for i in range(self.net_pars.depth):
            # extend with a fully-connected linear layer
            self.fc_layers.extend([nn.Linear(width, self.net_pars.width)])
            if self.net_pars.parallel:
                self.fc_layers2.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers3.extend([nn.Linear(width, self.net_pars.width)])
                self.fc_layers4.extend([nn.Linear(width, self.net_pars.width)])
            width = self.net_pars.width
            # if desired, add batch normalisation
            if self.net_pars.batch_norm:
                self.fc_layers.extend([nn.BatchNorm1d(self.net_pars.width)])
                if self.net_pars.parallel:
                    self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
                    self.fc_layers4.extend([nn.BatchNorm1d(self.net_pars.width)])
            # add ELU units for non-linearity
            self.fc_layers.extend([nn.ELU()])
            if self.net_pars.parallel:
                self.fc_layers2.extend([nn.ELU()])
                self.fc_layers3.extend([nn.ELU()])
                self.fc_layers4.extend([nn.ELU()])
            # if dropout is desired, add dropout regularisation
            if self.net_pars.dropout != 0:
                self.fc_layers.extend([nn.Dropout(self.net_pars.dropout)])
                if self.net_pars.parallel:
                    self.fc_layers2.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers3.extend([nn.Dropout(self.net_pars.dropout)])
                    self.fc_layers4.extend([nn.Dropout(self.net_pars.dropout)])
        # Final layer yielding output, with either 3 (fix S0) or 4 outputs of a single network, or 1 output
        # per network in case of parallel networks.
        if self.net_pars.parallel:
            self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(self.net_pars.width, self.outs))
            self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, self.outs))
            self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, self.outs))
            if self.net_pars.fitS0:
                self.encoder4 = nn.Sequential(*self.fc_layers4, nn.Linear(self.net_pars.width, self.outs))
        else:
            self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(self.net_pars.width, self.est_pars * self.outs))

    def forward(self, X):
        # select constraint method
        if self.net_pars.con == 'sigmoid':
            # define constraints
            Dmin = self.net_pars.cons_min[0]
            Dmax = self.net_pars.cons_max[0]
            fmin = self.net_pars.cons_min[1]
            fmax = self.net_pars.cons_max[1]
            Dpmin = self.net_pars.cons_min[2]
            Dpmax = self.net_pars.cons_max[2]
            S0min = self.net_pars.cons_min[3]
            S0max = self.net_pars.cons_max[3]
            # this network constrains the estimated parameters between two values by taking the sigmoid.
            # Advantage is that the parameters are constrained and that the mapping is unique.
            # Disadvantage is that the gradients go to zero close to the prameter bounds.
            params1 = self.encoder(X)
            # if parallel again use each param comes from a different output
            if self.net_pars.parallel:
                params2 = self.encoder2(X)
                params3 = self.encoder3(X)
                if self.net_pars.fitS0:
                    params4 = self.encoder4(X)
        elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
            if self.net_pars.con == 'abs':
                # this network constrains the estimated parameters to be positive by taking the absolute.
                # Advantage is that the parameters are constrained and that the derrivative of the function remains
                # constant. Disadvantage is that -x=x, so could become unstable.
                params1 = torch.abs(self.encoder(X))
                if self.net_pars.parallel:
                    params2 = torch.abs(self.encoder2(X))
                    params3 = torch.abs(self.encoder3(X))
                    if self.net_pars.fitS0:
                        params4 = torch.abs(self.encoder4(X))
            else:
                # this network is not constraint
                params1 = self.encoder(X)
                if self.net_pars.parallel:
                    params2 = self.encoder2(X)
                    params3 = self.encoder3(X)
                    if self.net_pars.fitS0:
                        params4 = self.encoder4(X)
        else:
            raise Exception('the chose parameter constraint is not implemented. Try ''sigmoid'', ''none'' or ''abs''')
        X_temp=[]
        for aa in range(self.outs):
            if self.net_pars.con == 'sigmoid':
                # applying constraints
                if self.net_pars.parallel:
                    Dp = Dpmin + torch.sigmoid(params1[:, aa].unsqueeze(1)) * (Dpmax - Dpmin)
                    Dt = Dmin + torch.sigmoid(params2[:, aa].unsqueeze(1)) * (Dmax - Dmin)
                    Fp = fmin + torch.sigmoid(params3[:, aa].unsqueeze(1)) * (fmax - fmin)
                    if self.net_pars.fitS0:
                        S0 = S0min + torch.sigmoid(params4[:, aa].unsqueeze(1)) * (S0max - S0min)
                else:
                    Dp = Dpmin + torch.sigmoid(params1[:, aa * self.est_pars + 0].unsqueeze(1)) * (Dpmax - Dpmin)
                    Dt = Dmin + torch.sigmoid(params1[:, aa * self.est_pars + 1].unsqueeze(1)) * (Dmax - Dmin)
                    Fp = fmin + torch.sigmoid(params1[:, aa * self.est_pars + 2].unsqueeze(1)) * (fmax - fmin)
                    if self.net_pars.fitS0:
                        S0 = S0min + torch.sigmoid(params1[:, aa * self.est_pars + 3].unsqueeze(1)) * (S0max - S0min)
            elif self.net_pars.con == 'none' or self.net_pars.con == 'abs':
                if self.net_pars.parallel:
                    Dp = params1[:, aa].unsqueeze(1)
                    Dt = params2[:, aa].unsqueeze(1)
                    Fp = params3[:, aa].unsqueeze(1)
                    if self.net_pars.fitS0:
                        S0 = params4[:, aa].unsqueeze(1)
                else:
                    Dp = params1[:, aa * self.est_pars + 0].unsqueeze(1)
                    Dt = params1[:, aa * self.est_pars + 1].unsqueeze(1)
                    Fp = params1[:, aa * self.est_pars + 2].unsqueeze(1)
                    if self.net_pars.fitS0:
                        S0 = params1[:, aa * self.est_pars + 3].unsqueeze(1)
            # the central voxel will give the estimates of D, f and D*. In all other cases a is always 0.
            if aa == 0:
                if self.supervised == False:
                    Dpout = copy.copy(Dp)
                    Dtout = copy.copy(Dt)
                    Fpout = copy.copy(Fp)
                    if self.net_pars.fitS0:
                        S0out = copy.copy(S0)
                else:
                    Dpout = Dp
                    Dtout = Dt
                    Fpout = Fp
                    if self.net_pars.fitS0:
                        S0out = S0
            # here we estimate X, the signal as function of b-values given the predicted IVIM parameters. Although
            # this parameter is not interesting for prediction, it is used in the loss function
            # in this a>0 case, we fill up the predicted signal of the neighbouring voxels too, as these are used in
            # the loss function.
            if self.net_pars.fitS0:
                X_temp.append(S0 * (Fp * torch.exp(-self.bvalues * Dp) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))
            else:
                X_temp.append((Fp * torch.exp(-self.bvalues * Dp) + (1 - Fp) * torch.exp(-self.bvalues * Dt)))
        X = torch.cat(X_temp,dim=1)
        if self.net_pars.fitS0:
            return X, Dtout, Fpout, Dpout, S0out
        else:
            return X, Dtout, Fpout, Dpout, torch.ones(len(Dtout))
