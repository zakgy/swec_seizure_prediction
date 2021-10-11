import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
from scipy.signal import stft
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from swec_utils import training_model, Results, load_data, load_data_partitioned, STFTDataset, numpy_to_torch_dtype_dict
from resnet import resnet34

MODEL_TYPE = 'CNN'
WINDOW_SIZE = [5, 10, 15, 30, 45, 60]
PIL = [1800, 3600]
SPH = [0]
I_DISTANCE = [0, 3600*24]

# basic CNN
class CNN(nn.Module):
    def __init__(self, input_size, conv_channels=(3, 10), kernel=(5, 5), dropout=0.0):
        super().__init__()
        num_flattened = (((input_size[1] - kernel[0] + 1) // 2 - kernel[1] + 1) // 2) * (((input_size[2] - kernel[0] + 1) // 2 - kernel[1] + 1) // 2) * conv_channels[1]
        self.conv1 = nn.Conv2d(input_size[0], conv_channels[0], kernel[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel[1])
        self.fc1 = nn.Linear(num_flattened, 120) # 5: 58, 7: 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.dropout1d = nn.Dropout(p=dropout)
        self.dropout2d = nn.Dropout2d(p=dropout)
        self.batchnorm1 = nn.BatchNorm2d(input_size[0])
        self.batchnorm2 = nn.BatchNorm2d(conv_channels[0])

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout2d(x)
        # print(x.shape)
        # x = self.dropout2d(x)
        x = self.batchnorm2(x)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        # x = self.dropout2d(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTM(nn.Module):
    def __init__(self, n_classes, n_features, seq_len, n_hidden, n_layers=1, dropout=0):
        super(LSTM, self).__init__()
        self.name = 'LSTM'

        self.n_classes = n_classes
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=self.n_hidden, num_layers=self.n_layers, batch_first=False, dropout=dropout)
        self.linear1 = nn.Linear(self.n_hidden, self.n_classes)
        self.linear2 = nn.Linear(self.seq_len, 1)
        self.dropout = nn.Dropout(dropout)

        self.hidden_cell = None

    def init_hidden(self, x, device, dtype):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden, dtype=dtype)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden, dtype=dtype)
        return [t.to(device) for t in (h0, c0)]
        
    def forward(self, x):

        self.hidden_cell = self.init_hidden(x, x.device, x.dtype)
        # (batch, n_features, seq_len) -> (seq_len, batch, n_hidden)
        lstm_out, self.hidden_cell = self.lstm(x.view(self.seq_len, -1, self.n_features), self.hidden_cell)

        # (seq_len, batch, n_hidden) -> (seq_len, batch, n_classes)
        lin_out = self.linear1(lstm_out)
        lin_out = self.dropout(lin_out)

        out = self.linear2(lin_out.view(-1, self.n_classes, self.seq_len))
        # out = lin_out[-1]
        # out = torch.mean(lin_out, dim=0)
        return out[:,:,0]

# u and s must be vectors of length data.shape[1]
def standardize(data, u, s):
    std_data = np.zeros(data.shape, dtype=np.float16)
    for i in range(data.shape[1]):
        std_data[:,i] = (data[:,i] - u[i]) / s[i]
    return std_data

u, s = [], []

def compute_u_s(data):
    for i in range(data.shape[1]):
        u.append(np.mean(data[:,i].astype(np.float64)))
        s.append(np.std(data[:,i].astype(np.float64)))
    return u, s

def pre_process(data, fs, dtype=np.float64, std=False):
    nperseg = 256 # hardcoded
    data = np.transpose(data, (0, 2, 1))
    stft_size = (129, data.shape[2] // (nperseg // 2) + 1)
    Zxx = np.zeros((data.shape[0], data.shape[1], stft_size[0], stft_size[1]), dtype=dtype)

    for i in range(data.shape[1]):      
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, Zxx[:, i, :] = stft(data[:,i,:], fs=fs)

    if (std):
        data = standardize(data, u, s)
    return Zxx

def grid_search(epochs, device, weights, train_loader, val_loader, dtype, kernel_sizes, n_filters, dropouts, save_path, plot_results=False, verbose=False):
    sensitivity = []
    specificity = []
    auc = []
    f1 = []
    pr_auc = []
    for k in kernel_sizes:
        for f in n_filters:
            for d in dropouts:
                if (verbose):
                    print(f"Training model with kernel size = {k}, filter_sizes = {f}, dropout = {d}.")
                input_size = next(iter(train_loader))[0].shape[1:]
                if (MODEL_TYPE == 'CNN'):
                    model = CNN(input_size=input_size, conv_channels=f, kernel=(k, k), dropout=d)
                if (MODEL_TYPE == 'RESNET'):
                    model = resnet34(input_channels=input_size[0], num_classes=2, dropout=0.5)
                if (MODEL_TYPE =='LSTM'):
                    print('lstm')
                    model = LSTM(n_classes=2, n_features=next(iter(train_loader))[0].shape[1], seq_len=next(iter(train_loader))[0].shape[2], n_hidden=20, n_layers=2, dropout=0.1)
                model.to(numpy_to_torch_dtype_dict[dtype])
                t_res, v_res = training_model(model, epochs, device, weights, train_loader, val_loader, dtype, plot_results=plot_results, verbose=verbose)
                sensitivity.append(v_res.sen[-1])
                specificity.append(v_res.spe[-1])
                auc.append(v_res.auc[-1])
                f1.append(v_res.f1[-1])
                pr_auc.append(v_res.pr_auc[-1])
                if (save_path):
                    print("Saving model")
                    save_path += 'CNN_k' + str(k) + '_f' + str(f[0]) + '_' + str(f[1]) + '_d' + str(d).replace('.', '')
                    torch.save(model.state_dict(), save_path)
    
    return t_res, v_res

def write_results_to_excel(writer, sheet_name, tr_res, vl_res):
    df = pd.DataFrame(data={'Sensitivity': [tr_res.sen[-1], vl_res.sen[-1]], 
        'Specificity': [tr_res.spe[-1], vl_res.spe[-1]], 
        'ROC AUC': [tr_res.auc[-1], vl_res.auc[-1]], 
        'F1 Score': [tr_res.f1[-1], vl_res.f1[-1]], 
        'PR AUC': [tr_res.pr_auc[-1], vl_res.pr_auc[-1]]
        }, index=['Training', 'Validation'])
    df.to_excel(writer, sheet_name)

# high level function to evaluate different data-based hyperparameters
def tune(subject_path, file_name, window_size, pil, sph, i_distance, partitioned=True, std=False, device='cpu', dtype=np.float64):
    with pd.ExcelWriter(file_name) as writer:
        if (MODEL_TYPE == 'LSTM'):
            epochs = 100
        else:
            epochs = 250
        k, f, dr = 5, (5, 5), 0.5
        for n in window_size:
            for p in pil:
                for s in sph:
                    for d in i_distance:
                        batch_size = 512 // (n // 5)
                        ds = 10 - min(9, d//(24*3600)) # sketchy formula to get a reasonable number of interictal segments
                        print(f"Sequence duration: {n} | PIL: {p} | SPH: {s} | Interictal distance: {d}")
                        if (partitioned):
                            num_test_sz = 1
                            tr_data, vl_data, ts_data, tr_labels, vl_labels, ts_labels, fs = load_data_partitioned(subject_path, n, s, p, d, ds=ds, dtype=dtype, num_test_sz=num_test_sz)
                        else:
                            data, labels, fs = load_data(subject_path, n, s, p, d, ds=ds, dtype=dtype)
                            tr_data, ts_data, tr_labels, ts_labels = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=0)
                            tr_data, vl_data, tr_labels, vl_labels = train_test_split(tr_data, tr_labels, test_size=0.2, stratify=tr_labels, random_state=0)
                        
                        if (MODEL_TYPE == 'CNN' or MODEL_TYPE == 'RESNET'):
                            if (std):
                                compute_u_s(tr_data)
                            tr_data = pre_process(tr_data, fs, dtype, std)
                            vl_data = pre_process(vl_data, fs, dtype, std)
                            ts_data = pre_process(ts_data, fs, dtype, std)
                        if (MODEL_TYPE == 'LSTM'):
                            tr_data = np.transpose(tr_data, (0, 2, 1))
                            vl_data = np.transpose(vl_data, (0, 2, 1))
                            ts_data = np.transpose(ts_data, (0, 2, 1))

                        weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(tr_labels), y=tr_labels), dtype=numpy_to_torch_dtype_dict[dtype], device=device)
                        train_dataset = STFTDataset(tr_data, tr_labels, dtype=dtype)
                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
                        val_dataset = STFTDataset(vl_data, vl_labels, dtype=dtype)
                        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

                        tr_res, vl_res = grid_search(epochs, device, weights, train_loader, val_loader, dtype, [k], [f], [dr], save_path='', plot_results=False)
                        
                        sheet_name = 'n' + str(n) + ' p' + str(p) + ' s' + str(s) + ' d' + str(d)
                        write_results_to_excel(writer, sheet_name, tr_res, vl_res)

path = './swec/'
if (len(sys.argv) > 1):
    path = sys.argv[1]
if (len(sys.argv) > 2):
    MODEL_TYPE = sys.argv[2]
subject = 'ID12'
subject_path = path + subject + '/' + subject
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
tune(subject_path, 'partitioned.xlsx', WINDOW_SIZE, PIL, SPH, I_DISTANCE, True, False, device, np.float16)
