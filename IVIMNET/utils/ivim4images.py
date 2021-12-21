import numpy as np


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
