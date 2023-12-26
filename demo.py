import os
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from MY_HDCA_module import train_HDCA, test_HDCA
from loadmat import load_data_and_labels
from AUC import auc

windowlength = 10

# 设置被试列表
subject_list = ["sub1A"]

# 初始化性能指标列
result_list = []


# 循环处理每个被试的数据
for subject in subject_list:
    # 构建数据文件和标签文件路径
    data_file_path = os.path.join("Data/qinghuamat/data", f"data_{subject}.mat")
    label_file_path = os.path.join("Data/qinghuamat/label", f"label_{subject}.mat")
    print(f"Loading subject {subject}...")
    # 在这里添加代码来加载MAT文件并处理数据
    data_need, label_need = load_data_and_labels(data_file_path, label_file_path)
    print(f"{subject} loaded and processed.")
    # 设置五折交叉验证
    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds)

    auc_list = []
    acc_list = []
    pre_list = []
    rec_list = []
    fpr_list = []
    BA_list = []
    F1_list = []
    kappa_list = []

    for fold, (train_index, test_index) in enumerate(skf.split(data_need, label_need)):
        # 根据索引划分训练和测试集
        train_data = data_need[train_index]
        train_label = label_need[train_index]
        test_data = data_need[test_index]
        test_label = label_need[test_index]

        # train训练过程
        threshold, w, c, S = train_HDCA(train_data, train_label, windowlength)

        # test测试过程
        Pd, pf, temp_acc, temp_pre, temp_rec,temp_fpr, temp_BA, temp_F1, temp_kappa = test_HDCA(threshold, w, c, test_data, test_label, windowlength, S)
        # 计算AUC
        AUC = auc(pf,Pd)

        # 将性能指标添加到列表
        auc_list.append(AUC)
        acc_list.append(temp_acc)
        pre_list.append(temp_pre)
        rec_list.append(temp_rec)
        fpr_list.append(temp_fpr)
        BA_list.append(temp_BA)
        F1_list.append(temp_F1)
        kappa_list.append(temp_kappa)
        print(f"Performance for {subject} computed. AUC: {AUC:.4f}, Accuracy: {temp_acc:.4f}, Precision: {temp_pre:.4f}, Recall: {temp_rec:.4f}, Fpr: {temp_fpr:.4f} BA: {temp_BA:.4f}, F1: {temp_F1:.4f}, Kappa: {temp_kappa:.4f}")

    # 计算性能指标的平均值和标准差
    auc_mean = np.mean(auc_list)
    acc_mean = np.mean(acc_list)
    pre_mean = np.mean(pre_list)
    rec_mean = np.mean(rec_list)
    fpr_mean = np.mean(fpr_list)
    BA_mean = np.mean(BA_list)
    F1_mean = np.mean(F1_list)
    kappa_mean = np.mean(kappa_list)

    auc_std = np.std(auc_list)
    acc_std = np.std(acc_list)
    pre_std = np.std(pre_list)
    rec_std = np.std(rec_list)
    fpr_std = np.std(fpr_list)
    BA_std = np.std(BA_list)
    F1_std = np.std(F1_list)
    kappa_std = np.std(kappa_list)

    # 添加性能指标到结果列表
    result_list.append({
        "Subject": subject,
        "mean auc": auc_mean,
        "mean acc": acc_mean,
        "mean rec": rec_mean,
        "mean pre": pre_mean,
        "mean fpr":fpr_mean,
        "mean BA": BA_mean,
        "mean F1": F1_mean,
        "mean kappa": kappa_mean,
        "std auc": auc_std,
        "std acc": acc_std,
        "std value": rec_std,
        "std pre": pre_std,
        "std fpr": fpr_std,
        "std BA": BA_std,
        "std F1": F1_std,
        "std kappa": kappa_std
    })

# 创建DataFrame以保存结果
result_df = pd.DataFrame(result_list)

# 将结果保存为 Excel 文件
output_path = "result.xlsx"
result_df.to_excel(output_path, index=False)
