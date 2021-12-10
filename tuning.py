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
from swec_utils import training_model, Results, load_data_partitions, STFTDataset, numpy_to_torch_dtype_dict, print_results
from resnet import resnet34
from swec_models import CNN, LSTM

seed = 0
MODEL_TYPE = 'CNN'
WINDOW_SIZE = [10, 15, 30, 60]
PIL = [1800, 3600, 7200]
I_DISTANCE = [0]
SPH = 0

TEST_FOLD = -1

def cross_validation(folds, epochs, batch_size, device, dtype=np.float64):
    t_res, v_res = Results(), Results()
    for k in range(len(folds)):
        print("Fold " + str(k))
        tr_data = np.concatenate([x[0] for i,x in enumerate(folds) if i!=k], axis=0)
        tr_labels = np.concatenate([x[1] for i,x in enumerate(folds) if i!=k], axis=0)
        vl_data = folds[k][0]
        vl_labels = folds[k][1]

        weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(tr_labels), y=tr_labels), dtype=numpy_to_torch_dtype_dict[dtype], device=device)
        train_dataset = STFTDataset(tr_data, tr_labels, dtype=dtype)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        val_dataset = STFTDataset(vl_data, vl_labels, dtype=dtype)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        kernel, filters, dropout = 5, (5, 5), 0.0
        input_size = next(iter(train_loader))[0].shape[1:]
        torch.manual_seed(seed)
        if (MODEL_TYPE == 'CNN'):
            model = CNN(input_size=input_size, conv_channels=filters, kernel=(kernel, kernel), dropout=dropout)
        elif (MODEL_TYPE == 'RESNET'):
            model = resnet34(input_channels=input_size[0], num_classes=2, dropout=dropout)

        model.to(numpy_to_torch_dtype_dict[dtype])
        t_res_temp, v_res_temp = training_model(model, epochs, device, weights, train_loader, val_loader, dtype, lr=0.0001, plot_results=False, verbose=False)

        t_res.add(t_res_temp)
        v_res.add(v_res_temp)
    t_res.divide(len(folds))
    v_res.divide(len(folds))

    return t_res, v_res, model

def pre_process(data, fs, dtype=np.float64):
    nperseg = 256 # hardcoded
    data = np.transpose(data, (0, 2, 1))
    stft_size = (nperseg // 2 + 1, data.shape[2] // (nperseg // 2) + 1)
    Zxx = np.zeros((data.shape[0], data.shape[1], stft_size[0], stft_size[1]), dtype=dtype)

    for i in range(data.shape[1]):      
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, Zxx[:, i, :] = stft(data[:,i,:], fs=fs, nperseg=nperseg)
    return Zxx

def write_results_to_excel(writer, sheet_name, tr_res, vl_res):
    df = pd.DataFrame(data={'Sensitivity': [tr_res.sen, vl_res.sen], 
        'Specificity': [tr_res.spe, vl_res.spe], 
        'ROC AUC': [tr_res.auc, vl_res.auc], 
        'F1 Score': [tr_res.f1, vl_res.f1], 
        'PR AUC': [tr_res.pr_auc, vl_res.pr_auc]
        }, index=['Training', 'Validation'])
    df.to_excel(writer, sheet_name)

# high level function to evaluate different data-based hyperparameters
def tune(subject_path, file_name, window_size, pil, i_distance, leaveoneout=True, device='cpu', dtype=np.float64):
    with pd.ExcelWriter(file_name) as writer:
        epochs = 100
        k, f, dr = 5, (5, 5), 0.5
        for n in window_size:
            for p in pil:
                for d in i_distance:
                    batch_size = 128
                    ds = 10 - min(9, d//(24*3600) + 1) # sketchy formula to get a reasonable number of interictal segments
                    print(f"Model: {MODEL_TYPE} | Sequence duration: {n} | PIL: {p} | Interictal distance: {d}")

                    shuffle = not leaveoneout
                    all_data, fs = load_data_partitions(subject_path, n, p, SPH, d, shuffle=shuffle, ds=ds, dtype=dtype, verbose=False)

                    all_data.pop(TEST_FOLD)

                    folds = all_data
                    # ts_data, ts_labels = all_data[-1]

                    for i in range(len(folds)):
                        folds[i][0] = pre_process(folds[i][0], fs, dtype=dtype)
                    
                    tr_res, vl_res, _ = cross_validation(folds, epochs=epochs, batch_size=batch_size, device=device, dtype=dtype)

                    print_results(tr_res, "Training")
                    print_results(vl_res, "Validation")
                    
                    sheet_name = 'n' + str(n) + ' p' + str(p) + ' d' + str(d)
                    write_results_to_excel(writer, sheet_name, tr_res, vl_res)

path = './swec/'
subject = 'ID12'
if (len(sys.argv) > 1):
    path = sys.argv[1]
if (len(sys.argv) > 2):
    subject = sys.argv[2]
if (len(sys.argv) > 3):
    MODEL_TYPE = sys.argv[3]
if (len(sys.argv) > 4):
    TEST_FOLD = int(sys.argv[4])

subject_path = path + subject + '/' + subject
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
excel_filename = 'leaveoneout_crossval_' + subject + '_' + MODEL_TYPE + '.xlsx'
if (TEST_FOLD != -1):
    excel_filename = 'leaveoneout_crossval_' + subject + '_' + MODEL_TYPE + '_' + str(TEST_FOLD) +  '.xlsx'
print("Patient: " + subject)
tune(subject_path, excel_filename, WINDOW_SIZE, PIL, I_DISTANCE, True, device, np.float16)
