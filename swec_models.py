import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from torchvision import models
from torch.nn.utils import weight_norm

########
# CNN  #
########
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

########
# LSTM #
########
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

########
# TCN  #
########
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (batch_size, num_features, seq_length)"""
        y1 = self.tcn(inputs)  # input should have dimension (batch_size, num_features, seq_length)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

