from scipy.io import loadmat
import numpy as np
import os

def load_data_and_labels(data_folder, labels_folder):
    data_need = loadmat(data_folder)['data_need']
    label_need = loadmat(labels_folder)['label_need']
    return data_need, label_need