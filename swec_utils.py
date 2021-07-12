import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import decimate
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_recall_curve, auc

FILE_DURATION = 3600

# dict to convert numpy dtypes to torch dtypes
numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
}

# convert tensor to numpy array
def to_numpy(x):
    if (x.is_cuda):
        return x.cpu().detach().numpy()
    else:
        return x.detach().numpy()

# counts number of segments in one EEG clip
def num_segments(data, seq_len, overlap=0):
    if (overlap >= seq_len):
        print("ERROR: overlap must be less than segment length")
        return -1
    count = 0
    i = 0
    while(i + seq_len <= data.shape[1]):
        i += seq_len - overlap
        count += 1
    return count

# splits one EEG clip into segments of specifed length
def sliding_window(data, seq_len, overlap=0, d=1, dtype=np.float64):
    n_seg = num_segments(data, seq_len, overlap)
    segments = np.zeros((n_seg, data.shape[0], seq_len//d), dtype=dtype)

    idx = 0
    i = 0
    while(i + seq_len <= data.shape[1]):
        new_segment = np.reshape(data[:, i:i+seq_len], (1, data.shape[0], seq_len))
        if (d > 1):
            new_segment = decimate(new_segment, d, axis=2)
        segments[idx] = new_segment
        i += seq_len - overlap
        idx += 1
    
    segments = np.transpose(segments, (0, 2, 1))
    return segments

# takes time stamp in seconds and finds the file index and offset
def find_file_index(t, fs):
    file_index = (t/FILE_DURATION + 1)[0].astype(int)
    offset = ((t%FILE_DURATION)*fs)[0].astype(int)
    return file_index, offset
    
# extract preictal segments from swec data
def get_preictal(subject_path, seizure_begin, seq_len, fs, sph, pil, channels=[0], overlap=0, dtype=np.float64):
    preictal_begin = seizure_begin - (sph + pil)
    file_index, offset = find_file_index(preictal_begin, fs)

    preictal_eeg = loadmat(subject_path + '_' + str(file_index) + 'h.mat')['EEG']
    n_files = math.ceil(pil/FILE_DURATION)
    for i in range(n_files):
        eeg = loadmat(subject_path + '_' + str(file_index + 1 + i) + 'h.mat')['EEG']
        preictal_eeg = np.concatenate((preictal_eeg, eeg), axis=1)
    preictal_eeg = preictal_eeg[:, offset:offset+pil*fs]
    preictal_segments = sliding_window(preictal_eeg, seq_len, overlap=overlap, dtype=dtype)
    preictal_segments = preictal_segments[:,:,channels]
    return preictal_segments

# return true if time point is not in the designated interictal period
def in_seizure_bounds(file_index, seizure_begin, seizure_end, distance):
    for i in range(len(seizure_begin)):
        lo_bound = seizure_begin[i] - distance
        hi_bound = seizure_end[i] + distance
        # print(lo_bound, hi_bound)
        lo_time_stamp = (file_index - 1) * FILE_DURATION
        hi_time_stamp = file_index * FILE_DURATION

        if (lo_time_stamp > lo_bound and lo_time_stamp < hi_bound):
            return True
        if (hi_time_stamp > lo_bound and hi_time_stamp < hi_bound):
            return True
    return False

# extract interictal segments from swec data
def get_interictal(subject_path, seizure_begin, seizure_end, seq_len, fs, distance, channels=[0], dtype=np.float64):
    if (len(seizure_begin) != len(seizure_end)):
        print("ERROR: seizure_begin and seizure_end must be same length.")
        return

    file_index = 1
    file_index_list = []
    while(True):
        if (not os.path.exists(subject_path + '_' + str(file_index) + 'h.mat')):
            if (len(file_index_list) == 0):
                print("ERROR: no interictal segments selected.")
            if (file_index_list[-1] == file_index-1):
                file_index_list.pop()
            break
        if (not in_seizure_bounds(file_index, seizure_begin, seizure_end, distance)):
            file_index_list.append(file_index)
        file_index += 1

    eeg_file_len = 3600 * fs
    interictal_eeg = np.zeros((len(channels), (len(file_index_list) * eeg_file_len)), dtype=dtype)
    
    for i, idx in enumerate(file_index_list):
        print(idx)
        eeg = loadmat(subject_path + '_' + str(idx) + 'h.mat')['EEG']
        interictal_eeg[:, eeg_file_len*i:eeg_file_len*i + eeg_file_len] = eeg[channels,:]
    interictal_segments = sliding_window(interictal_eeg, seq_len, dtype=dtype)
    return interictal_segments

# load swec data
def load_data(subject_path, seq_duration, sph, pil, distance, channels=[0], dtype=np.float64):
    # load info file
    data_info = loadmat(subject_path + '_info.mat')
    seizure_begin = data_info['seizure_begin']
    seizure_end = data_info['seizure_end']
    fs = data_info['fs'][0][0]

    print(seizure_begin)
    print(seizure_end)

    n_channels = loadmat(subject_path + '_' + str(1) + 'h.mat')['EEG'].shape[0]
    if (len(channels) == 0):
        channels = np.arange(n_channels)

    seq_len = seq_duration * fs

    preictal = np.zeros((len(seizure_begin) * pil // seq_duration, seq_len, len(channels)), dtype=dtype)
    i = 0
    for s in seizure_begin:
        preictal[i:i + (pil // seq_duration)] = get_preictal(subject_path, s, seq_len, fs, sph, pil, channels, dtype=dtype)
        i += (pil // seq_duration)
    print(preictal.shape)

    interictal = get_interictal(subject_path, seizure_begin, seizure_end, seq_len, fs, distance, channels, dtype=dtype)
    print(interictal.shape)

    labels = np.concatenate((np.zeros(len(interictal)), np.ones(len(preictal))))
    segments = np.concatenate((interictal, preictal), axis=0)

    return segments, labels, fs

# load swec data, where test data has a different seizure than train data
def load_data_partitioned(subject_path, seq_duration, sph, pil, distance, channels=[0], dtype=np.float64, num_test_sz=1):
    # load info file
    data_info = loadmat(subject_path + '_info.mat')
    seizure_begin = data_info['seizure_begin']
    seizure_end = data_info['seizure_end']
    fs = data_info['fs'][0][0]

    print(seizure_begin)
    print(seizure_end)

    n_channels = loadmat(subject_path + '_' + str(1) + 'h.mat')['EEG'].shape[0]
    if (len(channels) == 0):
        channels = np.arange(n_channels)

    seq_len = seq_duration * fs

    preictal_train = np.zeros(((len(seizure_begin) - num_test_sz) * pil // seq_duration, seq_len, len(channels)), dtype=dtype)
    preictal_test = np.zeros((num_test_sz * pil // seq_duration, seq_len, len(channels)), dtype=dtype)

    i = 0
    for s in range(len(seizure_begin) - num_test_sz):
        preictal_train[i:i + (pil // seq_duration)] = get_preictal(subject_path, seizure_begin[s], seq_len, fs, sph, pil, channels, dtype=dtype)
        i += (pil // seq_duration)
    print(preictal_train.shape)

    i = 0
    for s in range(len(seizure_begin) - num_test_sz, len(seizure_begin)):
        preictal_test[i:i + (pil // seq_duration)] = get_preictal(subject_path, seizure_begin[s], seq_len, fs, sph, pil, channels, dtype=dtype)
        i += (pil // seq_duration)
    
    interictal = get_interictal(subject_path, seizure_begin, seizure_end, seq_len, fs, distance, channels, dtype=dtype)
    print(interictal.shape)

    split_idx = int(len(interictal) * (1 - (num_test_sz / len(seizure_begin))))
    
    interictal_train = interictal[:split_idx]
    interictal_test = interictal[split_idx:]

    tr_labels = np.concatenate((np.zeros(len(interictal_train)), np.ones(preictal_train)))
    tr_segments = np.concatenate((interictal_train, preictal_train), axis=0)

    ts_labels = np.concatenate((np.zeros(len(interictal_test)), np.ones(preictal_test)))
    ts_segments = np.concatenate((interictal_test, preictal_test), axis=0)

    return tr_segments, tr_labels, ts_segments, ts_labels, fs

# Dataset class
class STFTDataset(Dataset):
    def __init__(self, X, Y, dtype=np.float64):
        self.data = X.astype(dtype)
        self.target = Y
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# Results class
class Results:
    def __init__(self, loss, sen, spe, auc, f1, pr_auc):
        self.loss = loss
        self.sen = sen
        self.spe = spe
        self.auc = auc
        self.f1 = f1
        self.pr_auc = pr_auc

def evaluate_performance(y_pred, y_true):
    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.float64)
    y_pred_pos_prob = F.softmax(torch.tensor(y_pred), dim=1)[:, 1]
    # TODO: change pos label to 0 and neg prediction
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_pos_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # roc_auc = roc_auc_score(y_true, y_pred_pos_prob)
    f1 = f1_score(y_true, np.argmax(y_pred, axis=1))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_pos_prob)
    pr_auc = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity, roc_auc, f1, pr_auc

# trains model and returns loss, sensitivity, specificity, auc, f1, and prauc
def train(model, device, train_loader, loss_func, optimizer, dtype):
    model.train()
    total_loss = 0
    pred_list = np.zeros((len(train_loader.dataset), 2), dtype=dtype)
    true_list = np.zeros(len(train_loader.dataset), dtype=dtype)
    i = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        pred_list[i:i+len(data)] = to_numpy(output)
        true_list[i:i+len(data)] = to_numpy(target)

        loss = loss_func(output, target.long())
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        i += len(data)
    
    sensitivity, specificity, roc_auc, f1, pr_auc = evaluate_performance(pred_list, true_list)
    return total_loss, sensitivity, specificity, roc_auc, f1, pr_auc

# validates model and returns loss, sensitivity, specificity, auc, f1, and prauc
def validate(model, device, val_loader, loss_func, dtype):
    model.eval()
    total_loss = 0
    pred_list = np.zeros((len(val_loader.dataset), 2), dtype=dtype)
    true_list = np.zeros(len(val_loader.dataset), dtype=dtype)
    i = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred_list[i:i+len(data)] = to_numpy(output)
            true_list[i:i+len(data)] = to_numpy(target)

            loss = loss_func(output, target.long())

            total_loss += loss.item()
            i += len(data)
    
    sensitivity, specificity, roc_auc, f1, pr_auc = evaluate_performance(pred_list, true_list)
    return total_loss, sensitivity, specificity, roc_auc, f1, pr_auc

# plot statistics from training
def plot_training(t_res, v_res):
    train_loss, train_sen, train_spe, train_auc = t_res.loss, t_res.sen, t_res.spe, t_res.auc
    val_loss, val_sen, val_spe, val_auc = v_res.loss, v_res.sen, v_res.spe, v_res.auc

    fig, ax = plt.subplots(4,1, figsize=(15,15))
    ax[0].plot(train_loss)
    ax[0].plot(val_loss)
    ax[0].legend(['Training', 'Validation'])
    ax[0].set_title('Loss')
    ax[1].plot(train_sen)
    ax[1].plot(val_sen)
    ax[1].legend(['Training', 'Validation'])
    ax[1].set_title('Sensitivity')
    ax[2].plot(train_spe)
    ax[2].plot(val_spe)
    ax[2].legend(['Training', 'Validation'])
    ax[2].set_title('Specificity')
    ax[3].plot(train_auc)
    ax[3].plot(val_auc)
    ax[3].legend(['Training', 'Validation'])
    ax[3].set_title('AUC')
    plt.show()

# full training of model
def training_model(model, epochs, device, weights, train_loader, val_loader, dtype, plot_results=False, verbose=True):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, eps=1e-04)

    train_loss = []
    train_sen = []
    train_spe = []
    train_auc = []
    train_f1 = []
    train_pr = []

    val_loss = []
    val_sen = []
    val_spe = []
    val_auc = []
    val_f1 = []
    val_pr = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        if (epoch % 10 == 0 and verbose):
            print(f"Epoch {epoch}.")
        
        loss_tr, sen_tr, spe_tr, auc_tr, f1_tr, pr_tr = train(model, device, train_loader, criterion, optimizer, dtype)
        train_loss.append(loss_tr)
        train_sen.append(sen_tr)
        train_spe.append(spe_tr)
        train_auc.append(auc_tr)
        train_f1.append(f1_tr)
        train_pr.append(pr_tr)

        loss_vl, sen_vl, spe_vl, auc_vl, f1_vl, pr_vl = validate(model, device, val_loader, criterion, dtype)
        val_loss.append(loss_vl)
        val_sen.append(sen_vl)
        val_spe.append(spe_vl)
        val_auc.append(auc_vl)
        val_f1.append(f1_vl)
        val_pr.append(pr_vl)

    print(f"Training:\n\tLoss: {train_loss[-1]}\n\tSensitivity: {train_sen[-1]}\n\tSpecificity: {train_spe[-1]}\n\tAUC: {train_auc[-1]}\n\tF1 Score: {train_f1[-1]}\n\tPR AUC: {train_pr[-1]}")
    print(f"Validation:\n\tLoss: {val_loss[-1]}\n\tSensitivity: {val_sen[-1]}\n\tSpecificity: {val_spe[-1]}\n\tAUC: {val_auc[-1]}\n\tF1 Score: {val_f1[-1]}\n\tPR AUC: {val_pr[-1]}")

    train_results = Results(train_loss, train_sen, train_spe, train_auc, train_f1, train_pr)
    val_results = Results(val_loss, val_sen, val_spe, val_auc, val_f1, val_pr)

    if(plot_results):
        plot_training(train_results, val_results)

    return train_results, val_results

