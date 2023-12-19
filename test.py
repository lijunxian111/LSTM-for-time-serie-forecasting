# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_preprocessing import read_files, read_files_v2, load_data
from model import Regression_Model, LSTM_model
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score

test_path = 'data/standard_1/test_2.csv'
method = 'xgb' #xgb, reg or time
use_cls = False

def test_deep_model(raw_data, full_data):
    X_test = torch.from_numpy(raw_data)[:, [0, 1, 2, 3, 4, 5, 9, 10, 11]]
    Y_test = raw_data[:, 8]
    #print(Y_test.shape)
    time_dim = X_test.shape[0]
    dim = X_test.shape[1]
    if method == 'reg':
        model = Regression_Model(dim, 128, 3)
    else:
        model = LSTM_model(dim, 64, 3)
    true_labels = (Y_test <= -27.0)
    print(true_labels)
    model.load_state_dict(torch.load(f'save_models/save_{method}.pt'), strict=False)
    # print(model.state_dict())
    if method == 'time':
        X_test = torch.reshape(X_test, shape=[time_dim, 1, dim])
    model.eval()
    results = model(X_test).detach().squeeze().numpy()
    #print(results)
    #print(np.sum(results[:, 2] <= -27.0))
    pred_labels = (results[:, 2] <= -27.0)
    print(f"总共预测的异常值有{np.sum(pred_labels)}个")
    cnt = 0.
    for i in range(len(pred_labels)):
        if true_labels[i] == pred_labels[i] and pred_labels[i] == True:
            cnt += 1
    print("与真实的异常位置对上的个数为:", cnt)
    print(f"准确率为{cnt / 1024.}%")
    dict = {'OLT接收光功率(dBm)': results[:,0],'发送光功率(dBm)': results[:,1] ,'接收光功率(dBm)': results[:,2]}
    df = pd.DataFrame(dict)
    df.to_csv('save_results/results_reg_xgb.csv')

def test_deep_model_v2(raw_data, full_data):
    X_test = torch.from_numpy(raw_data)[:, [0, 1, 2, 3, 4, 5]]
    Y_test = raw_data[:, 8]
    #print(Y_test.shape)
    time_dim = X_test.shape[0]
    dim = X_test.shape[1]
    if method == 'reg':
        model = Regression_Model(dim, 128, 2)
    else:
        model = LSTM_model(dim, 64, 2)
    true_labels = (Y_test <= -27.0)
    #print(true_labels)
    model.load_state_dict(torch.load(f'save_models/save_{method}_cls.pt'), strict=False)
    # print(model.state_dict())
    if method == 'time':
        X_test = torch.reshape(X_test, shape=[time_dim, 1, dim])
    model.eval()
    results = model(X_test).detach().squeeze().numpy()
    results = np.argmax(results, axis=1)
    #print(results)
    #print(np.sum(results[:, 2] <= -27.0))
    pred_labels = results
    print(f"总共预测的异常值有{np.sum(pred_labels)}个")
    cnt = 0.
    for i in range(len(pred_labels)):
        if true_labels[i] == pred_labels[i] and pred_labels[i] == True:
            cnt += 1
    print("与真实的异常位置对上的个数为:", cnt)
    print(f"准确率为{cnt / 1024.}%")
    dict = {'预测结果': pred_labels}
    df = pd.DataFrame(dict)
    df.replace(0, '正常', inplace=True)
    df.replace(1, '异常', inplace=True)
    df.to_csv('save_results/results.csv')

def test_xgb_model(raw_data, full_data):
    X_test = raw_data[:, [0, 1, 2, 3, 4, 5, 9, 10, 11]]
    Y_test = raw_data[:, 8]
    #Y_test = raw_data[:, 8]
    true_labels = (Y_test <= -27.0)
    #print(true_labels)
    #print(np.sum(true_labels))
    model = XGBRegressor(n_estimators=750, scale_pos_weight=2, max_depth=6)
    model.load_model('save_models/save_xgb.json')
    results = model.predict(X_test)
    pred_labels = (results[:, 2] <= -27.0)
    #pred_labels = results
    print(f"总共预测的异常值有{np.sum(pred_labels)}个")
    cnt = 0.
    for i in range(len(pred_labels)):
        if true_labels[i] == pred_labels[i] and pred_labels[i] == True:
            cnt += 1
    print("与真实的异常位置对上的个数为:",cnt)
    print(f"准确率为{cnt / 1024.}%")
    dict = {'OLT接收光功率(dBm)': results[:,0],'发送光功率(dBm)': results[:,1] ,'接收光功率(dBm)': results[:,2]}
    df = pd.DataFrame(dict)
    df.to_csv('save_results/results_reg_xgb.csv')


def test_xgb_model_v2(raw_data, full_data):
    X_test = raw_data[:, [0, 1, 2, 3, 4, 5]]
    Y_test = raw_data[:, 8]
    #Y_test = raw_data[:, 8]
    true_labels = (Y_test <= -27.0)
    #print(true_labels)
    #print(np.sum(true_labels))
    model = XGBClassifier(n_estimators=750, scale_pos_weight=2, max_depth=6)
    model.load_model('save_models/save_xgb_cls.json')
    results = model.predict(X_test)
    #print(results)
    #print(np.sum(results[:, 2] <= -27.0))
    #pred_labels = (results[:, 2] <= -27.0)
    pred_labels = results
    print(f"总共预测的异常值有{np.sum(pred_labels)}个")
    cnt = 0.
    for i in range(len(pred_labels)):
        if true_labels[i] == pred_labels[i] and pred_labels[i] == True:
            cnt += 1
    print("与真实的异常位置对上的个数为:",cnt)
    print(f"准确率为{cnt / 1024.}%")
    dict = {'预测结果': pred_labels}
    df = pd.DataFrame(dict)
    df.replace(0,'正常', inplace=True)
    df.replace(1, '异常', inplace=True)
    df.to_csv('save_results/results.csv')

if __name__ == "__main__":
    raw_data_1, full_data = read_files_v2(test_path)  #这里是读取那些固定字段的数据，作为特征用于预测
    #print(full_data)
    #print(raw_data_1)
    if method == 'xgb':
    #test_deep_model(raw_data_1)
        if use_cls:
            test_xgb_model_v2(raw_data_1, full_data) #这里用训练好的模型进行预测
        else:
            test_xgb_model(raw_data_1, full_data)
    else:
        if use_cls:
            test_deep_model_v2(raw_data_1, full_data)
        else:
            test_deep_model(raw_data_1, full_data)

    """
    模型预测准确率结果计算方法：
    总共1024个异常值，模型预测的异常值能有531个对照上，531/1024。
    """