import sys
import getopt
import numpy as np
import torch
import torch.nn as nn
from resnet import resnet18
import torch.nn.functional as F
from torchvision import transforms
import pickle
from scipy.signal import stft, cwt, ricker, decimate
import warnings
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from swec_utils import Results, load_data_partitions, training_model, STFTDataset, numpy_to_torch_dtype_dict, to_numpy, evaluate_performance, print_results

# model
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


# from numba import jit, njit
# import numba_scipy
# 
# @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
# def cwt2d(data, wavelet, widths):
#     output = np.zeros((data.shape[0], len(widths), data.shape[1]))
#     for i in range(len(data)):
#         cwt_mat = cwt(data[i], ricker, widths)
#         output[i] = cwt_mat
#     return output

tf_type = "CWT"
def pre_process(data, fs, dtype=np.float64):
    if (tf_type == "STFT"):
        nperseg = 1024 # hardcoded
        data = np.transpose(data, (0, 2, 1))
        stft_size = (nperseg // 2 + 1, data.shape[2] // (nperseg // 2) + 1)
        Zxx = np.zeros((data.shape[0], data.shape[1], stft_size[0], stft_size[1]), dtype=dtype)

        for i in range(data.shape[1]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, Zxx[:, i, :] = stft(data[:,i,:], fs=fs, nperseg=nperseg)
        return Zxx
        
    if (tf_type == "CWT"):
        downsample_factor = 16
        widths = np.arange(1, 31)
        Zxx = np.zeros((data.shape[0], data.shape[1], len(widths), data.shape[2]//downsample_factor), dtype=dtype)

        for i in range(data.shape[0]):
            start = time.time()
            for j in range(data.shape[1]):
                cwt_mat = cwt(data[i,j,:], ricker, widths, dtype=dtype)
                Zxx[i,j,:] = decimate(cwt_mat, downsample_factor, axis=1)
            end = time.time()
            print(i, end - start)
        return Zxx

def cross_validation(folds, epochs, batch_size, device, dtype=np.float64):
    t_res, v_res = Results(), Results()
    for k in range(len(folds)):
        print("Fold " + str(k))
        tr_data = np.concatenate([x[0] for i,x in enumerate(folds) if i!=k], axis=0)
        tr_labels = np.concatenate([x[1] for i,x in enumerate(folds) if i!=k], axis=0)
        vl_data = folds[k][0]
        vl_labels = folds[k][1]

        print(tr_data.shape)
        print(tr_labels.shape)
        print(vl_data.shape)
        print(vl_labels.shape)

        weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(tr_labels), y=tr_labels), dtype=numpy_to_torch_dtype_dict[dtype], device=device)
        train_dataset = STFTDataset(tr_data, tr_labels, dtype=dtype)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = STFTDataset(vl_data, vl_labels, dtype=dtype)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        kernel, filters, dropout = 5, (5, 5), 0.5
        input_size = next(iter(train_loader))[0].shape[1:]
        model = CNN(input_size=input_size, conv_channels=filters, kernel=(kernel, kernel), dropout=dropout)

        model.to(numpy_to_torch_dtype_dict[dtype])
        t_res_temp, v_res_temp = training_model(model, epochs, device, weights, train_loader, val_loader, dtype, lr=0.0001, plot_results=False, verbose=False)

        t_res.add(t_res_temp)
        v_res.add(v_res_temp)
    t_res.divide(len(folds))
    v_res.divide(len(folds))

    return t_res, v_res, model

def main(argv):
    WINDOW_SIZE = 30
    PIL = 3600
    I_DISTANCE = 3600*24
    path = "./swec/"

    leaveoneout = True

    try:
        opts, args = getopt.getopt(argv,"p:f:w:l:d:s:",["patient=","path=","window=","pil=","distance=","shuffle"])
    except getopt.GetoptError:
        print('single.py -p <patient name> -f <data path> -w <window size> -l <PIL> -d <interictal distance>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p", "--patient"):
            subject = arg
        elif opt in ("-f", "--path"):
            path = arg
        elif opt in ("-w", "--window"):
            WINDOW_SIZE = int(arg)
        elif opt in ("-l", "--pil"):
            PIL = int(arg)
        elif opt in ("-d", "--distance"):
            I_DISTANCE = int(arg)
        elif opt in ("-s", "--shuffle"):
            leaveoneout = False

    subject_path = path + subject + '/' + subject

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    dtype=np.float16
    EPOCHS = 100
    BATCH_SIZE = 32

    ds = 10 - min(9, I_DISTANCE//(24*3600)) # sketchy formula to get a reasonable number of interictal segments

    print(f"Processing raw iEEG data with window size = {WINDOW_SIZE}, pil = {PIL}, interictal distance = {I_DISTANCE}")

    shuffle = not leaveoneout
    data, fs = load_data_partitions(subject_path, WINDOW_SIZE, PIL, I_DISTANCE, shuffle=shuffle, ds=ds, dtype=dtype, verbose=True)

    folds = data[0:-1]
    ts_data, ts_labels = data[-1]

    for i in range(len(folds)):
        folds[i][0] = pre_process(folds[i][0], fs, dtype=dtype)
    
    print(folds[0][0].shape)
    print(ts_data.shape)

    data_path = "./data/w30d0"
    # pickle.dump(data, open(data_path, 'wb'))

    t_res, v_res, model = cross_validation(folds, epochs=100, batch_size=32, device=device, dtype=dtype)

    print_results(t_res, "Training")
    print_results(v_res, "Validation")

    test_dataset = STFTDataset(ts_data, ts_labels, dtype=dtype)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # testing
    model.eval()

    pred_list = np.zeros((len(test_loader.dataset), 2))
    true_list = np.zeros(len(test_loader.dataset))
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred_list[i:i+len(data)] = to_numpy(output)
            true_list[i:i+len(data)] = to_numpy(target)
            i += len(data)

    y_true = true_list
    y_pred = pred_list

    sensitivity, specificity, roc_auc, f1_score, pr_auc = evaluate_performance(y_pred, y_true)

    print("Sensitivity: " + str(sensitivity))
    print("Specificity: " + str(specificity))
    print("AUC: " + str(roc_auc))
    print("F1 Score: " + str(f1_score))
    print("PR AUC: " + str(pr_auc))

if __name__ == "__main__":
   main(sys.argv[1:])
