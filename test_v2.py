# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_preprocessing import read_files, read_files_v3, read_files_pre
from model import Regression_Model, LSTM_model_v3, LSTM_model_v2
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def test_deep_model(raw_data, n_future, raw_keys): #n_future用于调节预测几天，不要超过总体天数的1/2, 为了保证效果好最好在总体天数的1/3左右，尽量与main_v2.py里面保持一致
    raw_data = raw_data.astype(np.float32)
    lenth = raw_data.shape[1]
    #print(lenth)
    X_test = torch.from_numpy(raw_data[:, -12:, :][:, np.newaxis, ...]) #取最后（总长-要预测天数）个数据，-1：-1表示取不到最后一个，原来要取的第一个往前推了一天
    initial = raw_data[:, -1, 2]
    initial_idxs = (initial <= -27.0)
    dim = raw_data.shape[2]
    model = LSTM_model_v2(dim, 128, 1)
    model.load_state_dict(torch.load(f'save_models/save_time_hid_128.pt'), strict=False) #加载训好的模型
    model.eval()
    total_batch = X_test.shape[0]
    seq_length = X_test.shape[2]
    batch_num = total_batch // 128
    results = []
    cnt = 0
    print(X_test.shape)
    for i in range(batch_num):
        X_slice = X_test[i*128: (i+1)*128, ...]
        #print(X_slice.shape)
        #_, res_slice = model(X_slice).detach().squeeze().numpy()
        for step in range(n_future):
            #print(step)
            res_slice = model(X_slice)
            if step == n_future - 1:
                results.append(res_slice[..., 2].squeeze().detach().cpu().numpy())
            else:
                X_slice = torch.cat([X_slice[..., 1:, :], res_slice.reshape([128, 1, 1, 6])], dim=2)
            #results.append(res_slice)
        cnt += 128
    if cnt < total_batch:
        left_data = X_test[cnt:total_batch, ...]
        for step in range(n_future):
            res_slice = model(left_data)
            if step == n_future - 1:
                results.append(res_slice[..., 2].squeeze().detach().cpu().numpy())
            else:
                left_data = torch.cat([left_data[..., 1:, :], res_slice.reshape([total_batch-cnt, 1, 1, 6])], dim=2)

    results = np.concatenate(results, axis=0)
    pred_labels = (results <= -27.0)
    pred_idx = np.where(results <= -27.0)
    print(f"预测共{np.sum(pred_labels)}个异常值")
    labels = np.zeros_like(results[:])
    labels[pred_idx] = 1
    #df2 = pd.read_csv('data/standard_1/test_1110.csv', encoding='utf-8', encoding_errors='ignore')
    #df2['接收光功率(dBm)'].replace('--', 0.0, inplace=True)
    true_data_dic = read_files_pre('data/standard_1/test_1110.csv', train=False)
    true_data = []
    true_samples = []
    keys = raw_keys
    idx = 0
    for key in keys:
        if key in true_data_dic:
            true_samples.append(idx)
            true_data.append(true_data_dic[key][np.newaxis, :])
            idx += 1
            #equal_keys.append(key)
    true_vals = np.concatenate(true_data, axis=0)[..., 2].squeeze()
    pred_labels = pred_labels[true_samples, ...]
    initial_idxs = initial_idxs[true_samples, ...]
    #true_vals = df2['接收光功率(dBm)'].values.astype(np.float32)
    true_labels = (true_vals <= -27.0)
    cnt = 0
    cnt2 = 0.
    for i in range(len(pred_labels)):
        if true_labels[i] == pred_labels[i] and (pred_labels[i] == True) and (initial_idxs[i] == False):
            cnt += 1
        if (true_labels[i] == True) and initial_idxs[i] == False:
            cnt2 += 1
    print("与真实的异常位置对上的个数为:", cnt)
    print(cnt2)
    print(f'准确率为{cnt/cnt2}')
    raw_keys = np.array(raw_keys)
    dict = {'网元IP': raw_keys[:, 0], '网元类型': raw_keys[:, 1].astype(np.int32),'框号':raw_keys[:, 2],'槽号':raw_keys[:, 3],'端口号':raw_keys[:, 4],'ONU ID': raw_keys[:, 5],'ONU TYPE': raw_keys[:, 6].astype(np.int32),'接收光功率(dBm)': results[:], '是否异常': labels}
    df = pd.DataFrame(dict)
    df['网元类型'].replace(1, 'MA5608T', inplace=True)
    df['网元类型'].replace(2, 'MA5680T', inplace=True)
    df['网元类型'].replace(3, 'MA5800-X17', inplace=True)
    df['网元类型'].replace(4, 'MA5800-X2', inplace=True)
    df['ONU TYPE'].replace(0, 'GPON', inplace=True)
    df['ONU TYPE'].replace(1, 'EPON', inplace=True)
    df['是否异常'].replace(1.0, '是', inplace=True)
    df['是否异常'].replace(0.0, '否', inplace=True)
    #print(df['网元类型'])

    df.to_csv('save_results/results_time_v2.csv', index=False)
    #保存为csv可以找到





if __name__ == "__main__":
    raw_data_1, raw_keys = read_files_v3(17, train=False)  #这里是读取那些固定字段的数据，作为特征用于预测
    print(raw_data_1)
    test_deep_model(raw_data_1, 3, raw_keys)
