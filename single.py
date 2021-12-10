import sys
import getopt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from resnet import resnet18, resnet34
import torch.nn.functional as F
from torchvision import transforms
import pickle
from scipy.signal import stft, cwt, ricker, decimate
import warnings
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from swec_utils import Results, load_data_partitions, train, training_model, STFTDataset, numpy_to_torch_dtype_dict, to_numpy, evaluate_performance, print_results
from swec_models import CNN, LSTM, TCN

seed = 0
model_type = "CNN"
TIME_FREQUENCY = True

tf_type = "STFT"
def pre_process(data, fs, dtype=np.float64):
    data = np.transpose(data, (0, 2, 1))
    if (tf_type == "STFT"):
        nperseg = 256 # hardcoded
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
        print(Zxx.shape)

        for i in range(data.shape[0]):
            start = time.time()
            for j in range(data.shape[1]):
                cwt_mat = cwt(data[i,j,:], ricker, widths, dtype=dtype)
                Zxx[i,j,:] = decimate(cwt_mat, downsample_factor, axis=1)
            end = time.time()
            print(i, end - start)
        return Zxx

def cross_validation(folds, epochs, batch_size, device, dtype=np.float64, n_folds=None):
    t_res, v_res = Results(), Results()
    if (len(folds) == 1):
        print("ERROR: not enough folds for cross-validation.")
    if (n_folds == None):
        n_folds = len(folds)
    for k in range(n_folds):
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
        torch.manual_seed(seed)
        if (model_type == "CNN"):
            model = CNN(input_size=input_size, conv_channels=filters, kernel=(kernel, kernel), dropout=dropout)
        elif (model_type == "RESNET"):
            model = resnet34(input_channels=input_size[0], num_classes=2, dropout=dropout)
        elif (model_type == "TCN"):
            model = TCN(input_size=input_size[0], output_size=2, num_channels=[50, 50, 50], kernel_size=3, dropout=0.5)

        model.to(numpy_to_torch_dtype_dict[dtype])
        t_res_temp, v_res_temp = training_model(model, epochs, device, weights, train_loader, val_loader, dtype, lr=0.0001, plot_results=False, verbose=False)

        t_res.add(t_res_temp)
        v_res.add(v_res_temp)
    t_res.divide(n_folds)
    v_res.divide(n_folds)

    return t_res, v_res, model

def train_model(folds, epochs, batch_size, device, dtype=np.float64):
    tr_data = np.concatenate([x[0] for i,x in enumerate(folds)], axis=0)
    tr_labels = np.concatenate([x[1] for i,x in enumerate(folds)], axis=0)

    print(tr_data.shape)
    print(tr_labels.shape)

    weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(tr_labels), y=tr_labels), dtype=numpy_to_torch_dtype_dict[dtype], device=device)
    train_dataset = STFTDataset(tr_data, tr_labels, dtype=dtype)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    kernel, filters, dropout = 5, (5, 5), 0.5
    input_size = next(iter(train_loader))[0].shape[1:]
    torch.manual_seed(seed)
    if (model_type == "CNN"):
        model = CNN(input_size=input_size, conv_channels=filters, kernel=(kernel, kernel), dropout=dropout)
    elif (model_type == "RESNET"):
        model = resnet34(input_channels=input_size[0], num_classes=2, dropout=0.5)

    model.to(numpy_to_torch_dtype_dict[dtype])

    lr=0.0001
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-04)

    model.to(device)
    for epoch in range(1, epochs + 1):
        if (epoch % 10 == 0):
            print(f"Epoch {epoch}.")
        
        loss_tr, sen_tr, spe_tr, auc_tr, f1_tr, pr_tr = train(model, device, train_loader, criterion, optimizer, dtype)
    
    tr_results = Results(loss_tr, sen_tr, spe_tr, auc_tr, f1_tr, pr_tr)
        
    return tr_results, model

def main(argv):
    TEST_FOLD = -1
    WINDOW_SIZE = 30
    PIL = 3600
    SPH = 0
    I_DISTANCE = 0
    path = "./swec/"

    leaveoneout = True

    try:
        opts, args = getopt.getopt(argv,"p:f:w:l:d:s:m:t:",["patient=","path=","window=","pil=","distance=","sph=", "model=", "fold="])
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
        elif opt in ("-s", "--sph"):
            SPH = int(arg)
        elif opt in ("-m", "--model"):
            model_type = arg
        elif opt in ("-t", "--fold"):
            TEST_FOLD = int(arg)

    subject_path = path + subject + '/' + subject

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    dtype=np.float16
    EPOCHS = 100
    BATCH_SIZE = 128

    ds = 10 - min(9, I_DISTANCE//(24*3600) + 1) # sketchy formula to get a reasonable number of interictal segments

    print(f"Processing raw iEEG data with fold = {TEST_FOLD}, window size = {WINDOW_SIZE}, pil = {PIL}, sph = {SPH}, interictal distance = {I_DISTANCE}")

    shuffle = not leaveoneout
    all_data, fs = load_data_partitions(subject_path, WINDOW_SIZE, PIL, SPH, I_DISTANCE, shuffle=shuffle, ds=ds, dtype=dtype, verbose=True)

    if (TEST_FOLD == -1):
        TEST_FOLD = len(all_data) - 1
        print(TEST_FOLD)
    ts_data, ts_labels = all_data.pop(TEST_FOLD)
    folds = all_data

    if (TIME_FREQUENCY):
        for i in range(len(folds)):
            # print(np.unique(folds[i][1], return_counts=True))
            folds[i][0] = pre_process(folds[i][0], fs, dtype=dtype)
    else:
        for i in range(len(folds)):
            folds[i][0] = np.transpose(folds[i][0], (0, 2, 1))
            folds[i][0] = decimate(folds[i][0], 10, axis=2)
            print(folds[i][0].shape)

    # t_res, v_res, model = cross_validation(folds, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device, dtype=dtype)
    t_res, model = train_model(folds, epochs=EPOCHS, batch_size=BATCH_SIZE, device=device, dtype=dtype)

    print_results(t_res, "Training")
    # print_results(v_res, "Validation")

    print(ts_data.shape)
    print(np.unique(ts_labels, return_counts=True))

    if (TIME_FREQUENCY): 
        ts_data = pre_process(ts_data, fs, dtype=dtype)
    else: 
        ts_data = np.transpose(ts_data, (0, 2, 1))
        ts_data = decimate(ts_data, 10, axis=2)
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
