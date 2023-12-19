# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Regression_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Regression_Model,self).__init__()
        self.fc1 = nn.Linear(in_dim,hidden_dim // 2)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.act4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.direct = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        h = x
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.act4(self.bn4(self.fc4(x)))
        x = self.classifier(x)
        #down = self.direct(h)
        #x = x+down
        return x

class LSTM_model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers = 3):
        super(LSTM_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        """
        self.conv = self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        """
        self.lstm1 = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        #self.lin1 = nn.Linear(self.hidden_dim, hidden_dim)
        #self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.lin = nn.Linear(self.hidden_dim, out_dim)
    def forward(self,x, future_dim):
        #x = x.permute(0,2,1)
        #x = self.conv(x)
        #x = x.permute(0,2,1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # Forward propagate LSTM
        xn, _ = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print(xn.shape)
        # Decode the hidden state of the last time step
        out = self.lin(xn[:, -1, :])  # 此处的-1说明我们只取RNN最后输出的那个hn
        return out

class LSTM_model_v2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers = 3, use_attr = True):
        super(LSTM_model_v2, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.use_attr = use_attr
        """
        self.conv = self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        """
        self.lstm1 = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        #self.lin1 = nn.Linear(self.hidden_dim, hidden_dim)
        #self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.lin = nn.Linear(self.hidden_dim, in_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, in_dim)
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.beta = nn.Parameter(torch.empty(size=(2 * self.hidden_dim, 1)))
        nn.init.xavier_uniform_(self.w.weight, gain=1.414)
        nn.init.xavier_uniform_(self.beta.data, gain=1.414)

    def attention(self, x):
        # x (batch_size, seq_len, input_size/hidden_size)
        x = self.w(x)  # bsi--->bst
        seq_len, size = x.shape[1], x.shape[2]

        x1 = x.repeat(1, 1, seq_len).view(x.shape[0], seq_len * seq_len, -1)
        x2 = x.repeat(1, seq_len, 1)
        cat_x = torch.cat([x1, x2], dim=-1).view(x.shape[0], seq_len, -1, 2 * size)  # b s s 2*size

        e = F.leaky_relu(torch.matmul(cat_x, self.beta).squeeze(-1))  # bss
        attention = F.softmax(e, dim=-1)  # b s s
        out = torch.bmm(attention, x)  # bss * bst ---> bst
        out = F.relu(out)
        return out

    def forward(self, x):
        #x = x.permute(0,2,1)
        #x = self.conv(x)
        #x = x.permute(0,2,1)
        B_size = x.shape[0]
        future_dim = x.shape[1]
        seq_len = x.shape[2]
        feat_dim = x.shape[3]
        x = x.reshape([B_size*future_dim, seq_len, feat_dim])
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # Forward propagate LSTM
        xn, _ = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print(xn.shape)
        #xn = xn.reshape([B_size, future_dim, seq_len, self.hidden_dim])
        # Decode the hidden state of the last time step
        if self.use_attr:
            xn = self.attention(xn)
        out_embeds = xn[..., -1, :]
        #out_embeds.reshape([B_size*future_dim, self.hidden_dim])
        out = self.lin(out_embeds)  # 此处的-1说明我们只取RNN最后输出的那个hn
        out = out.reshape([B_size, future_dim, self.in_dim])
        return out #self.lin2(xn[..., -1:, :])

class LSTM_model_v3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers = 3, use_attr = True):
        super(LSTM_model_v3, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.use_attr = use_attr
        """
        self.conv = self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        """
        self.lstm1 = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        #self.lin1 = nn.Linear(self.hidden_dim, hidden_dim)
        #self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.lin = nn.Linear(self.hidden_dim, in_dim)
        #self.lin2 = nn.Linear(self.hidden_dim, in_dim)




    def forward(self, x):
        #x = x.permute(0,2,1)
        #x = self.conv(x)
        #x = x.permute(0,2,1)
        B_size = x.shape[0]
        future_dim = x.shape[1]
        seq_len = x.shape[2]
        feat_dim = x.shape[3]
        x = x.reshape([B_size*future_dim, seq_len, feat_dim])
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # Forward propagate LSTM
        xn, _ = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        #print(xn.shape)
        #xn = xn.reshape([B_size, future_dim, seq_len, self.hidden_dim])
        # Decode the hidden state of the last time step
        #if self.use_attr:
            #xn = self.attention(xn)
        out_embeds = xn[..., -1, :]
        #out_embeds.reshape([B_size*future_dim, self.hidden_dim])
        out = self.lin(out_embeds)  # 此处的-1说明我们只取RNN最后输出的那个hn
        out = out.reshape([B_size, future_dim, self.in_dim])
        return out #self.lin2(xn[..., -1:, :])

