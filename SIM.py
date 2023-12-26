import numpy as np
from scipy.linalg import eigh, pinv,sqrtm

def SIM(x, m, iterN):
    T, C, N = x.shape  # C: channel number, N: Number of time points within a trial, T: Number of trials

    if m is None:
        m = C
    if iterN is None:
        iterN = 10

    xs = np.mean(x, axis=0)  # 64*25

    x0 = np.transpose(x, (1, 0, 2))
    xnew = np.transpose(x0, (0, 2, 1))

    xn = np.reshape(xnew, (C, N * T)) - np.repeat(xs, T, axis=1) # Normalize: 64*1700
   #ECovSig = np.dot(xs, xs.T) / N  # Signal covariance: 64*64
    ECovRes = np.dot(xn, xn.T) / (N * T)  # Noise covariance: 64*64
    ECovRes = (ECovRes + ECovRes.T) / 2  # Ensure symmetry

    for n in range(iterN):
        Wh = np.linalg.inv(sqrtm(ECovRes))  # the whitening matrix
        xs_tilde = np.dot(Wh, xs)
        ECovSig = np.dot(xs_tilde, xs_tilde.T) / N
        ECovSig = (ECovSig + ECovSig.T) / 2
        D, W = eigh(ECovSig)  # Eigendecomposition

        I = np.argsort(D)[::-1]  # Sort eigenvalues in descending order
        W = W[:, I[:m]].T

        A = np.dot(sqrtm(ECovRes), pinv(W))
        S = pinv(A)
        z = np.dot(W, xs_tilde)

        # 计算对数似然度
        # trace_term = np.trace(np.dot(np.linalg.inv(ECovRes), np.dot(xn, xn.transpose(0, 2, 1)))
        # loglike = -N/2 * (T * N * np.log(2 * np.pi) + T * np.log(det(ECovRes)) + 1 / N * trace_term)
        A_z_replicated = np.repeat(np.dot(A, z), T, axis=1)
        xn = np.reshape(xnew, (C, N * T)) - A_z_replicated
        ECovRes = np.dot(xn, xn.T) / (N * T)

    return A, S, z

    # 示例用法
    # A, S, z = SIM(x)








    # 示例用法
# A, S, z = SIM(x, m, iterN)
