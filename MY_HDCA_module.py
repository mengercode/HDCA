import numpy as np
from sklearn.linear_model import LogisticRegression
# from RunSingleLR import RunSingleLR
from SIM import SIM
from fisher import fisher
from findThreshold import findThreshold
from ROC import ROC


# import statsmodels.api as sm


def train_HDCA(data, label, windowlength):
    trail, channel, samplepoint = data.shape

    m = 25
    Data = np.transpose(data, (1, 2, 0))
    target_id = np.where(label == 1)[0]
    other_id = np.where(label == 2)[0]

    target_signal = Data[:, :, target_id]
    other_signal = Data[:, :, other_id]
    num = round(samplepoint / windowlength)
    S = []

    temp_target_signal = np.transpose(target_signal, (2, 0, 1))
    Iteration = 10
    A, S, z = SIM(temp_target_signal, m, Iteration)

    temp_target = []
    for iii in range(target_signal.shape[2]):
        channel = m
        temp_target.append(np.dot(S, target_signal[:, :, iii]))

    temp_othtarget = []
    for jjj in range(other_signal.shape[2]):
        temp_othtarget.append(np.dot(S, other_signal[:, :, jjj]))
    temp_target = np.array(temp_target)
    temp_othtarget = np.array(temp_othtarget)

    target_signal = np.array(temp_target)
    other_signal = np.array(temp_othtarget)
    target_signal = np.transpose(target_signal, (1, 2, 0))
    other_signal = np.transpose(other_signal, (1, 2, 0))

    tar_num = target_signal.shape[2]
    oth_num = other_signal.shape[2]
    w = np.zeros((channel, windowlength))
    Y_tar_sig = np.zeros((tar_num, windowlength))
    Y_oth_sig = np.zeros((oth_num, windowlength))
    w_t = np.zeros((m,10))
    for i in range(windowlength):
        tar_blockms = target_signal[:, num * i + 1:num * (i + 1), :]
        tar_blockms = np.mean(tar_blockms, axis=1)  # Take the mean along the second axis
        tar_reshape = tar_blockms.reshape(channel, tar_num)
        tar_blockms = np.expand_dims(tar_blockms, axis=1)

        oth_blockms = other_signal[:, num * i + 1:num * (i + 1), :]
        oth_blockms = np.mean(oth_blockms, axis=1)  # Take the mean along the second axis
        oth_reshape = oth_blockms.reshape(channel, oth_num)
        oth_blockms = np.expand_dims(oth_blockms, axis=1)

        w, T = fisher(tar_reshape, oth_reshape)
        M = np.squeeze(w)
        w_t[:,i]=M
        for t in range(tar_num):
            Y_tar_sig[t, i] = np.dot(M.T, tar_blockms[:, :, t])  # 正例特征为：正例数量*窗口

        for t in range(oth_num):
            Y_oth_sig[t, i] = np.dot(M.T, oth_blockms[:, :, t])

    temp_label = np.hstack((np.ones(tar_num), np.zeros(oth_num)))
    temp_data = np.vstack((Y_tar_sig, Y_oth_sig))
    temp1_data = temp_data.copy()

    for iWin in range(windowlength):
        temp1_data[:, iWin] = temp1_data[:, iWin] / np.std(temp1_data[:, iWin])

    # Fit logistic regression model
    lr = LogisticRegression(solver='liblinear')
    lr.fit(temp1_data, temp_label)
    c = lr.coef_[0]

    final_score = np.dot(temp_data, c)
    threshold, accuracy = findThreshold(final_score, temp_label)

    return threshold, w_t, c, S


# def SIM(data, m, Iteration):
#     # 实现SIM函数的代码
#     pass


# def findThreshold(final_score, temp_label):
#     # 实现寻找阈值的代码
#     pass


# #def fisher(tar_reshape, oth_reshape):
#     # 实现fisher函数的代码
#     pass


# def RunSingleLR(data, label, params):
#     # 实现RunSingleLR函数的代码
#     pass

# 示例用法
# threshold, w, c, S = train_HDCA(data, label, windowlength)
def test_HDCA(threshold, w, c, data, label, windowlength, S):
    trail, channel, samplepoint = data.shape

    Data = np.transpose(data, (1, 2, 0))
    target_id = np.where(label == 1)[0]
    other_id = np.where(label == 2)[0]

    target_signal = Data[:, :, target_id]
    other_signal = Data[:, :, other_id]

    temp_target = []
    for iii in range(target_signal.shape[2]):
        temp_target.append(np.dot(S, target_signal[:, :, iii]))

    temp_othtarget = []
    for jjj in range(other_signal.shape[2]):
        temp_othtarget.append(np.dot(S, other_signal[:, :, jjj]))

    target_signal = np.array(temp_target)
    other_signal = np.array(temp_othtarget)
    target_signal = np.transpose(target_signal, (1, 2, 0))
    other_signal = np.transpose(other_signal, (1, 2, 0))

    num = round(samplepoint / windowlength)
    tar_num = target_signal.shape[2]
    oth_num = other_signal.shape[2]
    Y_tar_sig = np.zeros((tar_num, windowlength))
    Y_oth_sig = np.zeros((oth_num, windowlength))

    for i in range(windowlength):
        tar_blockms = target_signal[:, num * i:num * (i + 1), :]
        tar_blockms = np.mean(tar_blockms, axis=1)  # 导联取平均部分
        tar_blockms = np.expand_dims(tar_blockms, axis=1)

        oth_blockms = other_signal[:, num * i:num * (i + 1), :]
        oth_blockms = np.mean(oth_blockms, axis=1)  # 导联取平均部分
        oth_blockms = np.expand_dims(oth_blockms, axis=1)

        for t in range(tar_num):
            Y_tar_sig[t, i] = np.dot(w[:, i].T, tar_blockms[:, :, t])  # 正例特征为：正例数量*窗口

        for t in range(oth_num):
            Y_oth_sig[t, i] = np.dot(w[:, i].T, oth_blockms[:, :, t])

    tar_final_score = np.dot(Y_tar_sig, c.T)
    oth_final_score = np.dot(Y_oth_sig, c.T)
    tar_final_score = np.expand_dims(tar_final_score, axis=1)  # 可能有问题
    oth_final_score = np.expand_dims(oth_final_score, axis=1)
    m, n = np.where(tar_final_score > threshold)
    TP = len(m)
    FN = tar_num - TP
    mm, nn = np.where(oth_final_score > threshold)
    FP = len(mm)
    TN = oth_num - FP

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fpr = FP / (FP + TN)
    F1 = 2 * ((precision * recall) / (precision + recall))
    BA = (recall + TN / (TN + FP)) / 2
    Po = (TP + TN) / (TP + TN + FP + FN)
    Pe = ((TP + FN) * (TP + FP) + (FN + TN) * (FP + FN)) / ((TP + FN + FP + TN) ** 2)
    kappa = (Po - Pe) / (1 - Pe)

    Pd, Pf = ROC(tar_final_score, oth_final_score)

    return Pd, Pf, accuracy, precision, recall, fpr, BA, F1, kappa
