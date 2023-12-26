from scipy.io import loadmat
import numpy as np
import mne
import os

def load_and_preprocess_data(data_file_path):
    X = loadmat(data_file_path)

    EEG_data_train = X['EEGdata1']  # You can choose 'EEGdata42' if needed
    class_labels_train = X['class_labels'][0]
    trigger_positions_train = X['trigger_positions'][0]

    EEG_data_test = X['EEGdata2']  # You can choose 'EEGdata2' if needed
    class_labels_test = X['class_labels'][1]
    trigger_positions_test = X['trigger_positions'][1]

    # Define event IDs based on class labels (1 for target, 2 for non-target)
    event_id = {'Target': 1, 'Non-Target': 2}

    ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
                'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '', '']
    ch_types = ['eeg'] * EEG_data_train.shape[0]
    sfreq = 250  # Update the sampling frequency if different
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    events_train = []
    events_test = []

    for i, (trigger_pos_train, class_label_train) in enumerate(zip(trigger_positions_train, class_labels_train)):
        events_train.append([trigger_pos_train, 0, event_id['Target' if class_label_train == 1 else 'Non-Target']])

    for j, (trigger_pos_test, class_label_test) in enumerate(zip(trigger_positions_test, class_labels_test)):
        events_test.append([trigger_pos_test, 0, event_id['Target' if class_label_test == 1 else 'Non-Target']])

    events_train = np.array(events_train, dtype=np.int32)
    events_test = np.array(events_test, dtype=np.int32)

    raw_train = mne.io.RawArray(EEG_data_train, info)
    raw_test = mne.io.RawArray(EEG_data_test, info)

    X_train_list = []
    Y_train_list = []

    X_test_list = []
    Y_test_list = []

    for event in events_train:
        start_sample = event[0]
        end_sample = start_sample + sfreq
        epoch_data_train = raw_train[:, start_sample:end_sample][0]
        label_train = event[2]
        X_train_list.append(epoch_data_train)
        Y_train_list.append(label_train)

    for event in events_test:
        start_sample = event[0]
        end_sample = start_sample + sfreq
        epoch_data_test = raw_test[:, start_sample:end_sample][0]
        label_test = event[2]
        X_test_list.append(epoch_data_test)
        Y_test_list.append(label_test)

    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    X_test = np.array(X_test_list)
    Y_test = np.array(Y_test_list)

    return X_train, Y_train, X_test, Y_test
