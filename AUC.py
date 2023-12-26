import numpy as np

def auc(pf, pd):
    pf_diff = np.diff(pf)
    value = np.abs(np.sum(pd[1:] * pf_diff))
    return value
