# SUPER IVIM-DC

Intra-voxel incoherent motion (IVIM) analysis of fetal lungs Diffusion-Weighted MRI (DWI) data shows potential in providing quantitative imaging bio-markers that reflect, indirectly, diffusion and pseudo-diffusion for non-invasive fetal lung maturation assessment. However, long acquisition times, due to the large number of different 'b-value' images required for IVIM analysis, precluded clinical feasibility.

We introduce SUPER-IVIM-DC a deep-neural-networks (DNN) approach which couples supervised loss with a data-consistency term to enable IVIM analysis of DWI data acquired with a limited number of b-values.

We demonstrated the added-value of SUPER-IVIM-DC over both classical and recent DNN approaches for IVIM analysis through numerical simulations, healthy volunteer study, and IVIM analysis of fetal lung maturation from fetal DWI data.

Our numerical simulations and healthy volunteer study show that SUPER-IVIM-DC estimates of the IVIM model parameters from limited DWI data had lower normalized root mean-squared error compared to previous DNN-based approaches. Further, SUPER-IVIM-DC estimates of the pseudo-diffusion fraction parameter from limited DWI data of fetal lungs correlate better with gestational age compared to both to classical and DNN-based approaches (0.555 vs. 0.463 and 0.310).

SUPER-IVIM-DC has the potential to reduce the long acquisition times associated with IVIM analysis of DWI data and to provide clinically feasible bio-markers for non-invasive fetal lung maturity assessment.

## Usage

Clone and install the package using `pip install super-ivim-dc`

We also have [a sample notebook on Google Colab](https://colab.research.google.com/drive/1aCXO0-EecTcxp9j48q54OGTOyHKKnB7r?usp=sharing). 

### Run training, generate `.pt` files

```
from super_ivim_dc.train import train

import numpy as np
working_dir: str = './working_dir'
super_ivim_dc_filename: str = 'super_ivim_dc'  # do not include .pt
ivimnet_filename: str = 'ivimnet'  # do not include .pt

bvalues = np.array([0,15,30,45,60,75,90,105,120,135,150,175,200,400,600,800])
snr = 10
sample_size = 100

train(
    SNR=snr, 
    bvalues=bvalues, 
    super_ivim_dc=True,
    ivimnet=True,
    work_dir=working_dir,
    super_ivim_dc_filename=super_ivim_dc_filename,
    ivimnet_filename=ivimnet_filename,
    verbose=False
)
```

This will create the following files:
- **ivimnet_init.json**, **super_ivim_dc_init.json** - contains the initial values used in the training
- **ivimnet.pt**, **super_ivim_dc.pt** - the pytorch models


### Generate a random signal and test the generated model

```
from super_ivim_dc.infer import test_infer

test_infer(
    SNR=snr,
    bvalues=bvalues,
    work_dir=working_dir,
    super_ivim_dc_filename=super_ivim_dc_filename,
    ivimnet_filename=ivimnet_filename,
    save_figure_to=None,  # if set to None, the figure will be shown in the notebook
    sample_size=sample_size,
)
```

### Generate signal

```
from super_ivim_dc.IVIMNET import simulations

IVIM_signal_noisy, Dt, f, Dp = simulations.sim_signal(
    SNR=snr, 
    bvalues=bvalues, 
    sims=sample_size
)

Dt, f, Dp = np.squeeze(Dt), np.squeeze(f), np.squeeze(Dp)
```

### Run inference on the generated signal

```
from super_ivim_dc.infer import infer_from_signal

Dp_ivimnet, Dt_ivimnet, Fp_ivimnet, S0_ivimnet = infer_from_signal(
    signal=IVIM_signal_noisy, 
    bvalues=bvalues,
    model_path=f"{working_dir}/{ivimnet_filename}.pt",
)

Dp_superivimdc, Dt_superivimdc, Fp_superivimdc, S0_superivimdc = infer_from_signal(
    signal=IVIM_signal_noisy, 
    bvalues=bvalues,
    model_path=f"{working_dir}/{super_ivim_dc_filename}.pt",
)
```

## Citation

If you find this code useful in your research, please consider citing:

```
@article{Korngut_Rotman_Afacan_Kurugol_Zaffrani-Reznikov_Nemirovsky-Rotman_Warfield_Freiman_2022, title={SUPER-IVIM-DC: Intra-voxel incoherent motion based fetal lung maturity assessment from limited DWI data using supervised learning coupled with data-consistency}, volume={13432}, DOI={10.1007/978-3-031-16434-7_71}, journal={Lecture Notes in Computer Science}, author={Korngut, Noam and Rotman, Elad and Afacan, Onur and Kurugol, Sila and Zaffrani-Reznikov, Yael and Nemirovsky-Rotman, Shira and Warfield, Simon and Freiman, Moti}, year={2022}, pages={743–752}}
```

[The paper is also available on ArXiv](https://arxiv.org/abs/2206.03820).
