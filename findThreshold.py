import numpy as np

def findThreshold(p, label):
    # Sort scores & labels, high to low
    indx = np.argsort(-p)

    # Binary labels {0, 1}
    label = label > 0
    Np = np.sum(label == 1)
    Nn = np.sum(label == 0)

    # Initialize AUC variables
    Az = 0  # Area Under the Curve
    N = Np + Nn  # Total Number
    tp = np.zeros(N + 1)  # True Positive
    fp = np.zeros(N + 1)  # False Positive
    d = np.zeros(N)  # Distance to Origin

    # Calculate fractions and distance at each threshold
    for i in range(N):
        tp[i + 1] = tp[i] + label[indx[i]] / Np
        fp[i + 1] = fp[i] + (~label[indx[i]]) / Nn
        Az = Az + (~label[indx[i]]) * tp[i + 1] / Nn
        d[i] = np.sqrt((1 - tp[i]) ** 2 + (0 - fp[i]) ** 2)

    adj_tpr = tp - fp

    # Min distance = best threshold
    threshIndex = np.argmin(d)
    threshold = p[indx[threshIndex]]

    # threshIndex = np.argmax(adj_tpr)
    # threshold = p[indx[threshIndex]]

    accuracy = Az

    return threshold, accuracy

