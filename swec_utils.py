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

SEED = 0
FILE_DURATION = 3600
POSTICTAL_LENGTH = 1800
LEAD_SEIZURE_DISTANCE = 3600 * 4 # 4 hours from the last seziure for lead seizure, taken from Shiao et al.

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
# https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap?page=1&tab=votes#tab-top
def in_seizure_bounds(file_index, seizure_begin, seizure_end, sph, pil, distance):
    for i in range(len(seizure_begin)):
        lo_bound = seizure_begin[i] - sph - pil - distance
        hi_bound = seizure_end[i] + POSTICTAL_LENGTH + distance
        lo_time_stamp = (file_index - 1) * FILE_DURATION
        hi_time_stamp = file_index * FILE_DURATION

        if (lo_bound <= hi_time_stamp and hi_bound >= lo_time_stamp):
            return True
    return False

# extract interictal segments from swec data
def get_interictal(subject_path, seizure_begin, seizure_end, seq_len, fs, sph, pil, distance, channels=[0], ds=1, dtype=np.float64):
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
        if (not in_seizure_bounds(file_index, seizure_begin, seizure_end, sph, pil, distance)):
            file_index_list.append(file_index)
        file_index += 1

    segs_per_file = (FILE_DURATION * fs // seq_len) // ds
    interictal_segments = np.zeros(((len(file_index_list) * segs_per_file), seq_len, len(channels)), dtype=dtype)
    
    np.random.seed(SEED)
    for i, idx in enumerate(file_index_list):
        eeg = loadmat(subject_path + '_' + str(idx) + 'h.mat')['EEG']
        segments = sliding_window(eeg[channels,:], seq_len, dtype=dtype)
        sampled_idxs = np.sort(np.random.choice(len(segments), len(segments) // ds, replace=False))
        interictal_segments[i*segs_per_file:i*segs_per_file+segs_per_file] = segments[sampled_idxs]
    return interictal_segments

def get_interictal_new(subject_path, seizure_begin, seq_len, fs, sph, pil, distance, channels=[0], ds=1, dtype=np.float64):
    interictal_end = seizure_begin - (sph + pil + distance)
    interictal_start = interictal_end - LEAD_SEIZURE_DISTANCE

    start_index, start_offset = find_file_index(interictal_start, fs)
    end_index, end_offset = find_file_index(interictal_end, fs)
    file_indices = np.arange(start_index, end_index+1, dtype=int)
    interictal = np.zeros((len(channels), (end_index - start_index + 1)*FILE_DURATION*fs), dtype=dtype)
    for i, idx in enumerate(file_indices):
        eeg = loadmat(subject_path + '_' + str(idx) + 'h.mat')['EEG']
        interictal[:, i*FILE_DURATION*fs:(i+1)*FILE_DURATION*fs] = eeg[channels, :]
    interictal = interictal[:, start_offset:(end_index - start_index)*FILE_DURATION*fs + end_offset]
    interictal_segments = sliding_window(interictal, seq_len, dtype=dtype)
    return interictal_segments

def load_data_partitions(subject_path, seq_duration, pil, sph, distance, channels=[], ds=1, dtype=np.float64, shuffle=False, verbose=False):
    np.random.seed(SEED)
    # load info file
    data_info = loadmat(subject_path + '_info.mat')
    seizure_begin = data_info['seizure_begin']
    seizure_end = data_info['seizure_end']
    fs = data_info['fs'][0][0]

    if (verbose):
        print(seizure_begin)
        print(seizure_end)

    n_channels = loadmat(subject_path + '_' + str(1) + 'h.mat')['EEG'].shape[0]
    if (len(channels) == 0):
        channels = np.arange(n_channels)

    to_remove = []
    for i in range(1, len(seizure_begin)):
        if (seizure_begin[i] < seizure_end[i-1] + LEAD_SEIZURE_DISTANCE):
            to_remove.append(i)
    for i in range(len(seizure_begin)):
        if (seizure_begin[i] - sph - pil < 0 and not i in to_remove):
            to_remove.append(i)

    if (verbose):
        print(to_remove)
    if (len(to_remove) > 0):
        valid_seizures = np.delete(seizure_begin, to_remove, axis=0)

    seq_len = seq_duration * fs

    preictal = np.zeros((len(valid_seizures), pil // seq_duration, seq_len, len(channels)), dtype=dtype)
    i = 0
    for s in valid_seizures:
        preictal[i] = get_preictal(subject_path, s, seq_len, fs, sph, pil, channels, dtype=dtype)
        i += 1
    if(shuffle):
        preictal = np.array_split(np.random.shuffle(np.concatenate((preictal[:]))), len(seizure_begin), axis=0)
    if (verbose):
        print(preictal.shape)

    interictal = get_interictal(subject_path, seizure_begin, seizure_end, seq_len, fs, sph, pil, distance, channels, ds=ds, dtype=dtype)
    # interictal = np.zeros((len(seizure_begin), LEAD_SEIZURE_DISTANCE // seq_duration, seq_len, len(channels)), dtype=dtype)
    # i = 0
    # for s in seizure_begin:
    #     interictal[i] = get_interictal_new(subject_path, s, seq_len, fs, sph, pil, distance, channels, ds, dtype)
    #     i += 1
    if(shuffle):
        interictal = np.random.shuffle(interictal)
    if (verbose):
        print(interictal.shape)
    interictal = np.array_split(interictal, len(valid_seizures), axis=0)
    

    data = []
    for n in range(len(valid_seizures)):
        labels = np.concatenate((np.zeros(len(interictal[n])), np.ones(len(preictal[n]))))
        segments = np.concatenate((interictal[n], preictal[n]), axis=0)
        data.append([segments, labels])
    return data, fs

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
    def __init__(self, loss=0, sen=0, spe=0, auc=0, f1=0, pr_auc=0):
        self.loss = loss
        self.sen = sen
        self.spe = spe
        self.auc = auc
        self.f1 = f1
        self.pr_auc = pr_auc
    
    def add(self, other):
        self.loss += other.loss[-1]
        self.sen += other.sen[-1]
        self.spe += other.spe[-1]
        self.auc += other.auc[-1]
        self.f1 += other.f1[-1]
        self.pr_auc += other.pr_auc[-1]
    
    def divide(self, scalar):
        self.loss /= scalar
        self.sen /= scalar
        self.spe /= scalar
        self.auc /= scalar
        self.f1 /= scalar
        self.pr_auc /= scalar    

def evaluate_performance(y_pred, y_true):
    y_pred = y_pred.astype(np.float64)
    y_true = y_true.astype(np.float64)
    try:
        y_pred_pos_prob = F.softmax(torch.tensor(y_pred), dim=1)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_pos_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1))
        precision, recall, _ = precision_recall_curve(y_true, y_pred_pos_prob)
        pr_auc = auc(recall, precision)
        tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    except ValueError:
        print("NAN or INF found.")
    except:
        print("Other error.")


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

def print_results(res, partition):
    print(partition + ":")
    print(f"\tLoss: {res.loss}\n\tSensitivity: {res.sen}\n\tSpecificity: {res.spe}\n\tAUC: {res.auc}\n\tF1 Score: {res.f1}\n\tPR AUC: {res.pr_auc}")

# full training of model
def training_model(model, epochs, device, weights, train_loader, val_loader, dtype, lr=0.0005, plot_results=False, verbose=True):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-04)

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

    if (verbose):
        print(f"Training:\n\tLoss: {train_loss[-1]}\n\tSensitivity: {train_sen[-1]}\n\tSpecificity: {train_spe[-1]}\n\tAUC: {train_auc[-1]}\n\tF1 Score: {train_f1[-1]}\n\tPR AUC: {train_pr[-1]}")
        print(f"Validation:\n\tLoss: {val_loss[-1]}\n\tSensitivity: {val_sen[-1]}\n\tSpecificity: {val_spe[-1]}\n\tAUC: {val_auc[-1]}\n\tF1 Score: {val_f1[-1]}\n\tPR AUC: {val_pr[-1]}")

    train_results = Results(train_loss, train_sen, train_spe, train_auc, train_f1, train_pr)
    val_results = Results(val_loss, val_sen, val_spe, val_auc, val_f1, val_pr)

    if(plot_results):
        plot_training(train_results, val_results)

    return train_results, val_results

