import numpy as np

def fisher(object_feature, nonobject_feature):
    # 计算两类样本的均值向量
    m1 = np.mean(object_feature, axis=1).reshape(-1, 1)
    m2 = np.mean(nonobject_feature, axis=1).reshape(-1, 1)

    # 样本向量减去均值向量
    temp1 = object_feature - m1
    temp2 = nonobject_feature - m2

    # 计算各类的类内离散度矩阵
    S1 = np.dot(temp1, temp1.T)
    S2 = np.dot(temp2, temp2.T)
    Sw = S1 + S2

    # 计算最佳投影方向向量
    w = np.linalg.pinv(Sw).dot(m1 - m2)

    T = -0.5 * (m1 + m2).T.dot(np.linalg.pinv(Sw)).dot(m1 - m2)

    return w, T

# # 示例用法
# object_feature = np.array([[1, 2, 3], [4, 5, 6]])  # 替换为你的目标特征
# nonobject_feature = np.array([[7, 8, 9], [10, 11, 12]])  # 替换为你的非目标特征
# w, T = fisher(object_feature, nonobject_feature)
# print("w:", w)
# print("T:", T)
