# SUPER IVIM-DC

Intra-voxel incoherent motion (IVIM) analysis of fetal lungs Diffusion-Weighted MRI (DWI) data shows potential in providing quantitative imaging bio-markers that reflect, indirectly, diffusion and pseudo-diffusion for non-invasive fetal lung maturation assessment. However, long acquisition times, due to the large number of different 'b-value' images required for IVIM analysis, precluded clinical feasibility.

We introduce SUPER-IVIM-DC a deep-neural-networks (DNN) approach which couples supervised loss with a data-consistency term to enable IVIM analysis of DWI data acquired with a limited number of b-values.

We demonstrated the added-value of SUPER-IVIM-DC over both classical and recent DNN approaches for IVIM analysis through numerical simulations, healthy volunteer study, and IVIM analysis of fetal lung maturation from fetal DWI data.

Our numerical simulations and healthy volunteer study show that SUPER-IVIM-DC estimates of the IVIM model parameters from limited DWI data had lower normalized root mean-squared error compared to previous DNN-based approaches. Further, SUPER-IVIM-DC estimates of the pseudo-diffusion fraction parameter from limited DWI data of fetal lungs correlate better with gestational age compared to both to classical and DNN-based approaches (0.555 vs. 0.463 and 0.310).

SUPER-IVIM-DC has the potential to reduce the long acquisition times associated with IVIM analysis of DWI data and to provide clinically feasible bio-markers for non-invasive fetal lung maturity assessment.

## Usage

Clone and install the package using `pip install .`

### Run simulation
Use as a python package:
```
from super_ivim_dc.simulate import simulate
simulate()
```

or use as a script: `super-ivim-dc-sim`

This will create a directory called `output` in your current working directory, which will contain the pytorch model of the simulation.
See super_ivim_dc/simulate.py for possible arguments

### Run inference
As a package:
```
from super_ivim_dc.infer import infer
infer()
```

## References

* Korngut, N., Rotman, E., Afacan, O., Kurugol, S.,  Zaffrani-Reznikov, Y., Nemirovsky-Rotman, S., Warfield, S., Freiman, M.: SUPER-IVIM-DC: Intra-voxel incoherent motion based Fetal lung maturity assessment from limited DWI data using supervised learning coupled with data-consistency, https://arxiv.org/abs/2206.03820 (2022)