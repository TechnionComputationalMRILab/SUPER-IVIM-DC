import numpy as np
import matplotlib.pyplot as plt
from Classsic_ivim_fit import IVIM_fit_sls_trf, fitMonoExpModel, IVIM_model

def ivimN_noS0(bvalues, Dt, Fp, Dp):
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp * np.exp(-bvalues * Dp ) + (1 - Fp) * np.exp(-bvalues * Dt))

def plot_corr(GA, f, mode, two_stage):

    Fp_corr1 = np.corrcoef(GA, f)

    plt.scatter(GA, f)

    if (two_stage):
        GA_less_26, GA_bigger_26 = GA[np.where(GA < 26)], GA[np.where(GA >= 26)]
        f_less_26, f_bigger_26 = f[np.where(GA < 26)], f[np.where(GA >= 26)]
        Fp_corr1, Fp_corr2  = np.corrcoef(GA_less_26,f_less_26), np.corrcoef(GA_bigger_26,f_bigger_26)
        m1, b1 = np.polyfit(GA_less_26, f_less_26, 1, rcond=None, full=False, w=None, cov=False)
        m2, b2 = np.polyfit(GA_bigger_26, f_bigger_26, 1, rcond=None, full=False, w=None, cov=False)
        line1 = f'y={b1:.2f}+{m1:.2f}x, r={Fp_corr1[0,1]:.3f}'
        line2 = f'y={b2:.2f}+{m2:.2f}x, r={Fp_corr2[0,1]:.3f}'
        plt.plot(GA_less_26, m1*(GA_less_26)+b1, label= 'Canalicular Stage')
        plt.plot(np.insert(GA_bigger_26,0, GA_less_26.max()), m2*(np.insert(GA_bigger_26,0, GA_less_26.max()))+b2, label = 'Saccular Stage')
    plt.xlabel('Gestational age in weeks', fontdict={'fontsize': 22})
    plt.ylabel('Fp' , fontdict={'fontsize': 22})#"ADC [mm^2/s]")
    if (mode == 'LS'):
        plt.ylim((0,0.5))
    else:
        plt.ylim((0.2,0.5))
    plt.title(f"{mode} (R="+str(format(Fp_corr1[0,1], ".3f")+')'), fontdict={'fontsize': 25})
    plt.legend(loc='upper right', prop={'size': 16})



def plot_IVIM_signal(S0, Dt, Fp, Dp, filename, bvalues, sig_vec, label):
    if  (label == 'SUPER-IVIM-DC' ):
        color = 'b'
    elif  (label == 'IVIMNET' ):
        color = 'orange'
    elif  (label == 'LS' ):
        color = 'green'
    plt.plot(bvalues, np.log(sig_vec),'o')
    bvals_plot =  np.arange(0,bvalues.max()+10,10)
    sls = IVIM_model(bvals_plot, Dt, Dp, Fp, 1)
    plt.plot(bvals_plot, np.log(sls), color, label = label)
    plt.title("Ivim Model for case " + (filename.replace('case', '')).replace('.nii',''))
    plt.xlabel("b values")
    plt.ylabel("log(S/s0)")
    plt.legend()

def ivim4images(b_val, D_star, D, Fp):
    """
    Create ivim model images based ob b values and parameters.
    :param b: list of different b values e.g. [0 10 20 30 40 50].
    :param D_star: image of D* values per pixel, shape of sx, sy
    :param D: image of D values per pixel, shape of sx, sy
    :param Fp: image of fraction values per pixel, shape of sx, sy
    return ivim_out: DWMRI images generated from D*, D, Fp parameters (num of images will be len(b))
    # fixe bug *** ValueError: operands could not be broadcast together with shapes (8,640,640) (8,)
    """
    sb = len(b_val)
    sx, sy = D.shape # to support clinical images with different sizes
    ivim_out = np.zeros(sb*sy*sx).reshape(sx,sy,sb)
    one_mat = np.ones(sx*sy).reshape(sx,sy)
    for i, b in enumerate(b_val):
        b_val_mat = b*one_mat
        #arg1 = (-b_val_mat)*(D_star+D)
        arg1 = (-b_val_mat)*(D_star+D)
        arg2 = (-b_val_mat)*D
        ivim = Fp*np.exp(arg1) + (1-Fp)*np.exp(arg2)
        ivim_out[:,:,i] = ivim
    #print(ivim_out)
    return ivim_out


def plot_IVIM_param(D_star, D, Fp, s0_mask = 1):
    """
    plot a figure of IVIM parameters map
    """
    fig, ax = plt.subplots(1, 3, figsize=(20,20))

    D_star_plot = ax[0].imshow(D_star*s0_mask, cmap='gray', clim=(0, 0.1))
    ax[0].set_title('D*')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.colorbar(D_star_plot, ax=ax[0], fraction=0.046, pad=0.04)

    D_plot = ax[1].imshow(D*s0_mask, cmap='gray', clim=(0, 0.004))#
    ax[1].set_title('D')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.colorbar(D_plot, ax=ax[1],fraction=0.046, pad=0.04)

    Fp_plot = ax[2].imshow(Fp*s0_mask, cmap='gray', clim=(0, 0.5))#
    ax[2].set_title('Fp')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    fig.colorbar(Fp_plot, ax=ax[2],fraction=0.046, pad=0.04)

    plt.subplots_adjust(hspace=-0.5)
    plt.show()
