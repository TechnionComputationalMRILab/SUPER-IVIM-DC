
def order(Dt, Fp, Dp, S0=None):
    # function to reorder D* and D in case they were swapped during unconstraint fitting. Forces D* > D (Dp>Dt)
    if Dp < Dt:
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp
    if S0 is None:
        return Dt, Fp, Dp
    else:
        return Dt, Fp, Dp, S0
