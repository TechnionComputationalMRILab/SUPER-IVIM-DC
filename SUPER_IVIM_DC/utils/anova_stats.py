import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM


def anova_stats(IVIM_Organ):
    test_df_melt = pd.melt(IVIM_Organ, value_vars=['IVIMNET', 'SUPER-IVIM', 'SUPER-IVIM-DC'])
    test_df_melt.columns = ['network', 'value']
    test_model = ols('value~C(network)', data=test_df_melt).fit()
    test_model.summary()
    anova = sm.stats.anova_lm(test_model, typ=1)
    print(anova)
    return anova


def anova_stats_reps_mea(organ_IVIM_DL1, organ_IVIM_DL2, organ_IVIM_DL3):
    organ_Net = pd.DataFrame(organ_IVIM_DL1, columns=["CV"]).assign(Network="IVIMNET")
    organ_Sup = pd.DataFrame(organ_IVIM_DL2, columns=["CV"]).assign(Network="SUPER-IVIM")
    organ_DC = pd.DataFrame(organ_IVIM_DL3, columns=["CV"]).assign(Network="SUPER-IVIM-DC")
    organ_AovaRM = pd.concat([organ_Net, organ_Sup, organ_DC]).reset_index()
    aovrm = AnovaRM(organ_AovaRM, 'CV', 'index', within=['Network'])
    result = aovrm.fit()

    print(result)
    print(f'median values on IVIM NET {organ_Net.median()}')
    print(f'median values on SUPER IVIM {organ_Sup.median()}')
    print(f'median values on SUPER IVIM DC {organ_DC.median()}')
    return aovrm, result


def anova_stats_reps_mea_sec(organ_IVIM_DL1, organ_IVIM_DL2, net2_name):
    organ_Net = pd.DataFrame(organ_IVIM_DL1, columns=["CV"]).assign(Network="IVIMNET")
    organ_net2 = pd.DataFrame(organ_IVIM_DL2, columns=["CV"]).assign(Network=net2_name)
    organ_AovaRM = pd.concat([organ_Net, organ_net2]).reset_index()
    aovrm = AnovaRM(organ_AovaRM, 'CV', 'index', within=['Network'])
    result = aovrm.fit()

    print(result)

    return aovrm, result
