# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_preprocessing import read_files_v3, load_time_data_v2, load_time_data_v3, data_aug_v2
from model import LSTM_model, LSTM_model_v2
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV


filepath = 'data/standard_1/ready_12.csv'
files = 17  #文件个数，有多少个文件就是多少
batch_size = 64
learning_rate = 0.001 #学习率，可以适当调参但是结果大差不差
epochs = 500
pretrain = True
patience = 50
hidden_dim = 128

#这个函数暂时没有用
def evaluate(epoch, model, data, loss):
    model = model.to('cpu')
    model.eval()
    total_mse = 0.
    cnt = 0.
    for step, (x, y) in enumerate(data):
        outputs = model(x)
        mse = loss(outputs, y)
        total_mse += mse.item()
        cnt += 1.0
    total_mse = total_mse / cnt
    print(f"epoch:{epoch}, eval_loss:{total_mse}")
    return total_mse

#用于模型训练的函数，一般不用动
def train_time(model, train_data, future_dim, optimizer, loss):
    best_mse = 1e9
    time = 0
    for e in tqdm(range(epochs)):
        model.train()
        cnt = 0.
        total_loss = 0.
        for step, (x,y) in enumerate(train_data):
            optimizer.zero_grad()
            #print(x.shape)
            outputs = model(x) #把数据和要预测的维度输入模型 #model(x, future_dim)
            #print(outputs)
            batch_loss = loss(outputs, y)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            cnt += 1.0
        total_loss = total_loss/cnt
        print(f"epoch:{e}, loss:{total_loss}")
        if total_loss < best_mse:
            best_mse = total_loss

            torch.save(model.state_dict(), f'save_models/save_time_hid_{hidden_dim}.pt')
            time = 0
        else:
            time += 1
            if time == patience:
                break

#用于跑深度模型的函数
def run_deep_v2(raw_data):
    train_set, dim, future_dim = load_time_data_v3(raw_data, batch_size=64, future=5)
    """
    这一句很重要！在这里改要预测的天数, future=
    """
    model = LSTM_model_v2(dim, hidden_dim, 1) #64是隐藏层维度，可以试试128会不会更好
    #这个参数一般不用改
    if pretrain:
        global epochs
        model.load_state_dict(torch.load(f'save_models/save_time_hid_{hidden_dim}.pt'), strict=False) #把训好的模型保存到相应路径
        epochs = 300
    loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters()) #优化器，不要动
    train_time(model, train_set, future_dim, optimizer, loss)


if __name__ == "__main__":
    raw_data, _ = read_files_v3(files, train=True) # files就是文件个数，在上面我指示的位置调
    print(raw_data.shape)
    #_, _, _ = load_time_data_v3(raw_data, batch_size)
    #raw_data = data_aug_v2(raw_data) #数据增强
    run_deep_v2(raw_data) #跑模型并且保存


