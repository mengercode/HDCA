import numpy as np

def ROC(y_target, y_other):
    amin = np.min(y_other)
    amax = np.max(y_other)
    middata = (amax - amin) / 1e5
    x = np.arange(amin, amax, middata)  # x 表示阈值选择范围
    Pd = np.zeros(len(x))
    Pf = np.zeros(len(x))

    for i in range(len(x)):
        Pd[i] = np.sum(y_target > x[i]) / len(y_target)
        Pf[i] = np.sum(y_other> x[i]) / len(y_other)



    return Pd, Pf

# 示例用法
# Pd, Pf = roc(Y_taris, Y_othis)
