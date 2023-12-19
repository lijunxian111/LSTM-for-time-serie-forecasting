# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import csv
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

## 读取文件数据的代码，注意要预先把.xlsx格式转化为.csv格式
def read_files(filepath):
    df = pd.read_csv(filepath,encoding='utf-8',encoding_errors='ignore')
    ## 读取文件，并且将某些属性抛弃，替换
    df.drop(columns=['网元IP','ONU TYPE'], inplace=True)
    df['网元类型'].replace('MA5608T',1,inplace=True)
    df['网元类型'].replace('MA5680T', 2, inplace=True)
    df['网元类型'].replace('MA5800-X17', 3, inplace=True)
    df['网元类型'].replace('MA5800-X2', 4, inplace=True)
    ## 将存在缺失值的属性栏目删除
    df.drop(df[df['ONU测距(m)']=='--'].index, inplace=True)
    df.drop(df[df['ONU测距(m)'] == '-'].index, inplace=True)
    df.drop(df[df['OLT接收光功率(dBm)'] == '-'].index, inplace=True)
    df.drop(df[df['OLT接收光功率(dBm)'] == '--'].index, inplace=True)
    df.drop(df[df['发送光功率(dBm)'] == '--'].index, inplace=True)
    df.drop(df[df['接收光功率(dBm)'] == '--'].index, inplace=True)
    df.drop(df[df['光模块电流(mA)'] == '-'].index, inplace=True)
    df.drop(df[df['光模块电压(V)'] == '-'].index, inplace=True)
    df.drop(df[df['光模块温度(°C)'] == '-'].index, inplace=True)
    X = df.values
    for lines in X:
        for i in range(len(lines)):
            lines[i] = float(lines[i])

    X = X.astype(np.float32)
    #print(X.max(axis=0))
    #print(X.min(axis=0))
    #X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
    #print(np.unique(X[:,1]))

    return X

def read_files_v2(filepath):
    df = pd.read_csv(filepath,encoding='utf-8',encoding_errors='ignore')
    ## 读取文件，并且将某些属性抛弃，替换
    #print(np.unique(df['网元IP'].values))
    df.drop(columns=['网元IP','ONU TYPE','框号','操作状态', '最后一次上线时间', '最后一次下线时间'], inplace=True)
    if '最后一次下线原因' in list(df.keys()):
        df.drop(columns=['最后一次下线原因'], inplace=True)
    df['网元类型'].replace('MA5608T',1,inplace=True)
    df['网元类型'].replace('MA5680T', 2, inplace=True)
    df['网元类型'].replace('MA5800-X17', 3, inplace=True)
    df['网元类型'].replace('MA5800-X2', 4, inplace=True)
    ## 将存在缺失值的属性栏目删除
    df['运行状态'].replace('在线', 1, inplace=True)
    df['运行状态'].replace('离线', 0, inplace=True)
    X_full = df.values
    df.drop(df[df['ONU测距(m)']=='--'].index, inplace=True)
    df.drop(df[df['ONU测距(m)'] == '-'].index, inplace=True)
    df.drop(df[df['OLT接收光功率(dBm)'] == '-'].index, inplace=True)
    df.drop(df[df['OLT接收光功率(dBm)'] == '--'].index, inplace=True)
    df.drop(df[df['发送光功率(dBm)'] == '--'].index, inplace=True)
    df.drop(df[df['接收光功率(dBm)'] == '--'].index, inplace=True)
    df.drop(df[df['光模块电流(mA)'] == '-'].index, inplace=True)
    df.drop(df[df['光模块电压(V)'] == '-'].index, inplace=True)
    df.drop(df[df['光模块温度(°C)'] == '-'].index, inplace=True)
    X = df.values
    for lines in X:
        for i in range(len(lines)):
            lines[i] = float(lines[i])

    X = X.astype(np.float32)
    #print(X.max(axis=0))
    #print(X.min(axis=0))
    #X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
    #print(np.unique(X[:,1]))

    return X, X_full

#用于预处理文件的函数, 这个函数一般不用动
def read_files_pre(file_path, train=True):
    """

    :param file_path: 输入的文件路径
    :return: 一个字典，对应每个光猫的属性
    """
    df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
    df.drop(columns=['操作状态','运行状态', '最后一次上线时间', '最后一次下线时间'], inplace=True) #删除一些不需要的变化字段
    if '最后一次下线原因' in list(df.keys()):
        df.drop(columns=['最后一次下线原因'], inplace=True)
    """
    将文字信息转为数字编码，可以直接输入神经网络
    """
    df['网元类型'].replace('MA5608T', 1, inplace=True)
    df['网元类型'].replace('MA5680T', 2, inplace=True)
    df['网元类型'].replace('MA5800-X17', 3, inplace=True)
    df['网元类型'].replace('MA5800-X2', 4, inplace=True)
    df['ONU TYPE'].replace('GPON', 0, inplace=True)
    df['ONU TYPE'].replace('EPON', 1, inplace=True)
    if train == False:
        df['ONU测距(m)'].replace('--', 0.0, inplace=True)
        df['ONU测距(m)'].replace('-', 0.0, inplace=True)
        df['OLT接收光功率(dBm)'].replace('-', 0.0, inplace=True)
        df['OLT接收光功率(dBm)'].replace('--', 0.0, inplace=True)
        df['发送光功率(dBm)'].replace('--', 0.0, inplace=True)
        df['接收光功率(dBm)'].replace('--', 0.0, inplace=True)
        df['光模块电流(mA)'].replace('-', 0.0, inplace=True)
        df['光模块电压(V)'].replace('-', 0.0, inplace=True)
        df['光模块温度(°C)'].replace('-', 0.0, inplace=True)
    else:

        df.drop(df[df['ONU测距(m)'] == '--'].index, inplace=True)
        df.drop(df[df['ONU测距(m)'] == '-'].index, inplace=True)
        df.drop(df[df['OLT接收光功率(dBm)'] == '-'].index, inplace=True)
        df.drop(df[df['OLT接收光功率(dBm)'] == '--'].index, inplace=True)
        df.drop(df[df['发送光功率(dBm)'] == '--'].index, inplace=True)
        df.drop(df[df['接收光功率(dBm)'] == '--'].index, inplace=True)
        df.drop(df[df['光模块电流(mA)'] == '-'].index, inplace=True)
        df.drop(df[df['光模块电压(V)'] == '-'].index, inplace=True)
        df.drop(df[df['光模块温度(°C)'] == '-'].index, inplace=True)

    X = df.values
    keys = X[:, 0:7]
    vals = X[:, 8:]
    dicts = {}
    #构造一个字典，结构为键：1-7的字段，值：后面的运行表现
    for i in range(len(vals)):
        for j in range(len(vals[i])):
            vals[i][j] = float(vals[i][j])
        raw_key = tuple(keys[i].tolist())
        dicts[raw_key] = vals[i]
    #f#or i in range(vals.shape[0]):
        #[keys[i]] = vals[i]
    return dicts

#用于做多个日期的文件读取并将信息整合的函数, 这个函数只需要改一个参数files——num, 在main_v2里面改
def read_files_v3(files_num = 16, train=True):
    """

    :param files_num: 这个参数是用来表示一共有多少天的文件的，16就代表16天
    :return:
    values: 形状为（光猫个数，天数，特征数）
    equal_keys: 用于标识唯一光猫的所有键, 无重复
    """
    now_dic = read_files_pre('data/standard_1/ready_13.csv', train=train)
    keys = now_dic.keys()
    values = np.array(list(now_dic.values()))[:, np.newaxis, :]
    equal_keys = []
    for i in range(13, files_num+12):
       raw_dic = read_files_pre(f'data/standard_1/ready_{i}.csv', train=train)
       val_ext = []
       #raw_keys = raw_dic.keys()
       equal_keys = []
       idx = 0
       true_samples = []
       for key in keys:
           if key in raw_dic:
                true_samples.append(idx)
                val_ext.append(raw_dic[key][np.newaxis, :])
                idx += 1
                equal_keys.append(key)
       val_ext = np.concatenate(val_ext, axis=0)[:, np.newaxis, :]
       #val_ext = np.array(list(raw_dic.values()))[true_samples, :][:, np.newaxis, :]
       values = np.concatenate([values[true_samples, :], val_ext], axis=1)
       keys = equal_keys
    """
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            for k in range(values.shape[2]):
                if values[i][j][k] == 0.0:
                    if j == 0:
                        values[i][j][k] = values[i][j+1][k]
                    elif 0<j and j<values.shape[1]-1:
                        values[i][j][k] = (values[i][j - 1][k] + values[i][j+1][k])/2
                    else:
                        values[i][j][k] = values[i][j-1][k]
    """
    return values, equal_keys

#用于改变预测天数的函数，这个函数只在main_v2.py里面用
def load_time_data_v2(X, batch_size, future = 5):
    """

    :param X:  传进来的原始特征
    :param batch_size: 一个批次的大小，一般不用动，用于训练
    :param future: 比较重要，作用是改变要预测的天数，比如预测十天就是future = 10
    调用方法： loaa_time_data_v2(X, 64, future = 要预测的天数)
    :return:
    """
    # Creating a data structure with 72 timestamps and 1 output
    X_train = []
    y_train = []
    X = X.astype(np.float32)
    lenth = X.shape[0]
    indexes = np.random.permutation(lenth)
    X = X[indexes, ...]
    #lenth = X.shape[1]
    n_future = future  # Number of days we want to predict into the future.
    n_past = X.shape[1] - n_future  # Number of past days we want to use to predict future.
    """
    for i in range(n_past, X.shape[0] - n_future + 1):
        X_train.append(training_set_scaled[i - n_past:i,
                       0:dataset_train.shape[1]])
        y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
    """
    """
    for i in range(0, n_future):
           X_train.append(torch.from_numpy(X[:, i:i+n_past, :]))
           y_train.append(torch.from_numpy(X[:,i+n_past,:]))
    """
    #X_train, y_train = np.array(X_train), np.array(y_train)
    X_train, y_train = torch.from_numpy(X[:, :n_past, :]),  torch.from_numpy(X[:, n_past:, 3:4])
    print('X_train shape == {}.'.format(X_train.shape))
    print('y_train shape == {}.'.format(y_train.shape))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    train_dim = X_train.shape[2]
    return train_loader, train_dim, n_future


def data_aug_v3(X: np.ndarray, y: np.ndarray):
    n_future = y.shape[1]
    initial_X = X
    print(X.shape[0])
    initial_y = y
    for i in range(n_future):
        idxes = np.where(initial_y[:, i, 2] <= -27.0)[0].tolist()
        print(len(idxes))
        pos_samples = initial_X[idxes, ...]
        #aug_samples = pos_samples
        pos_labels = initial_y[idxes]
        for _ in range(40):
            X = np.concatenate([X, pos_samples], axis=0)
            y = np.concatenate([y, pos_labels], axis=0)
    print('X_train shape == {} after data augmentation.'.format(X.shape))
    print('y_train shape == {} after data augmentation.'.format(y.shape))
    return X, y

def load_time_data_v3(X, batch_size, future = 5):
    X_train = []
    y_train = []
    X = X.astype(np.float32)
    lenth = X.shape[0]
    indexes = np.random.permutation(lenth)
    X = X[indexes, ...]
    # lenth = X.shape[1]
    n_future = future  # Number of days we want to predict into the future.
    n_past = X.shape[1] - n_future
    for i in range(0, n_future):
        X_train.append(X[:, i:i+n_past, :][:, np.newaxis, :, :])
        y_train.append(X[:, i+n_past:i+n_past+1, :])

    X_train = np.concatenate(X_train, axis=1)
    y_train = np.concatenate(y_train, axis=1)
    X_train, y_train = data_aug_v3(X_train, y_train)
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    print('X_train shape == {}.'.format(X_train.shape))
    print('y_train shape == {}.'.format(y_train.shape))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    train_dim = X_train.shape[3]
    return train_loader, train_dim, n_future

## 数据增强的函数，具体操作是，把小于-27.0的数据成倍复制加入训练
def data_aug(raw_data: np.ndarray):
    idxs = np.where(raw_data[:,8] <= -26.0)[0].tolist()

    neg_samples = raw_data[idxs,:]
    #print(neg_samples[:,8])
    aug_sample = neg_samples
    for i in range(80):
        neg_samples = np.concatenate([neg_samples, aug_sample], axis=0)

    raw_data = np.concatenate([raw_data, neg_samples], axis=0)
    #print(raw_data.shape)
    return raw_data

## 数据增强的函数，具体操作是，把小于-27.0的数据成倍复制加入训练
def data_aug_v2(raw_data: np.ndarray):
    lenth = raw_data.shape[1]
    idxs = np.where(raw_data[:,lenth-1, 3] <= -26.0)[0].tolist()

    neg_samples = raw_data[idxs, ...]
    #print(neg_samples[:,8])
    aug_sample = neg_samples
    for i in range(40):
        neg_samples = np.concatenate([neg_samples, aug_sample], axis=0)

    raw_data = np.concatenate([raw_data, neg_samples], axis=0)
    #print(raw_data.shape)
    return raw_data



def load_data(X, batch_size):
    X = X.astype(np.float32)
    lenth = X.shape[0]
    indexes = np.random.permutation(lenth)
    X = X[indexes, :]
    X_train = torch.from_numpy(X[:int(lenth * 0.7), [0,1,2,3, 4, 5, 9, 10, 11]])
    train_dim = X_train.shape[1]
    Y_train = torch.from_numpy(X[:int(lenth * 0.7), [6, 7, 8]])
    #print(Y_train)
    X_val = torch.from_numpy(X[int(lenth * 0.7):, [0,1,2,3, 4, 5, 9, 10, 11]])
    Y_val = torch.from_numpy(X[int(lenth * 0.7):, [6, 7, 8]])

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset,batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset,batch_size,shuffle=False, num_workers=2)
    return train_loader, val_loader, train_dim

def load_data_v2(X, batch_size):
    X = X.astype(np.float32)
    lenth = X.shape[0]
    indexes = np.random.permutation(lenth)
    X = X[indexes, :]
    X_train = torch.from_numpy(X[:int(lenth * 0.7), [0, 1, 2, 3, 4, 5]])
    train_dim = X_train.shape[1]
    Y_train = np.zeros((X_train.shape[0]))
    ill_indexs = np.where(X[:int(lenth * 0.7), 8] <= -27.0)
    # print(Y_train)
    Y_train[list(ill_indexs)] = 1
    Y_train = torch.from_numpy(Y_train).long()

    X_val = torch.from_numpy(X[int(lenth * 0.7):, [0, 1, 2, 3, 4, 5]])
    Y_val = np.zeros((X_val.shape[0]))
    ill_indexs_val = np.where(X[int(lenth * 0.7):, 8] <= -27.0)
    # print(Y_train)
    Y_val[list(ill_indexs_val)] = 1
    Y_val = torch.from_numpy(Y_val).long()

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_dim

def load_time_data(X, batch_size, tw = 50):
    inout_seq_train = []
    inout_lab_train = []
    inout_seq_val = []
    inout_lab_val = []
    L = X.shape[0]
    X_train = torch.from_numpy(X[:int(L * 0.7),:])
    X_val = torch.from_numpy(X[int(L * 0.7):,:])
    L_train = X_train.shape[0]
    L_val = X_val.shape[0]
    for i in range(L_train - tw):
        train_seq = X_train[i:i + tw, [0,1,2,3, 4, 5, 9, 10, 11]]
        train_label = X_train[i + tw:i + tw + 1, [6, 7, 8]]
        inout_seq_train.append(train_seq)
        inout_lab_train.append(train_label)
    for i in range(L_val - tw):
        val_seq = X_val[i:i + tw, [0,1,2,3, 4, 5, 9, 10, 11]]
        val_label = X_val[i + tw:i + tw + 1, [6, 7, 8]]
        inout_seq_val.append(val_seq)
        inout_lab_val.append(val_label)
    train_set = torch.stack(inout_seq_train, dim=0)
    train_dim = train_set.shape[-1]
    train_label = torch.stack(inout_lab_train, dim=0)
    val_set = torch.stack(inout_seq_val, dim=0)
    val_label = torch.stack(inout_lab_val, dim=0)
    train_dataset = TensorDataset(train_set, train_label)
    val_dataset = TensorDataset(val_set, val_label)
    """
    print(train_set.shape)
    print(train_label.shape)
    print(val_set.shape)
    print(val_label.shape)
    """
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_dim

def load_xgb_data(X):
    X = X.astype(np.float32)
    lenth = X.shape[0]
    indexes = np.random.permutation(lenth)
    X = X[indexes, :]
    x = X[:,[0,1,2,3, 4, 5, 9, 10, 11]]
    y = X[:,[6,7,8]]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def load_xgb_data_v2(X):
    X = X.astype(np.float32)
    lenth = X.shape[0]
    indexes = np.random.permutation(lenth)
    X = X[indexes, :]
    x = X[:,[0,1,2,3, 4, 5]]
    y = np.zeros(X.shape[0])
    ill_indexes = np.where(X[:,8] <= -27.0)
    y[list(ill_indexes)] = 1
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    #with open('data/standard_1/ready_2.csv', 'r', encoding='utf-8') as file:
        #reader = csv.reader(file)
        ##for row in reader:
            #print(row)
    #X = read_files('data/standard_1/ready_2.csv')
    #load_data(X, 64)
    #load_time_data(X,64)
    #A,B,C,D = load_xgb_data(X)
    #X = read_files_v2('data/standard_1/test_2.csv')
    V = read_files_v3(16)
    print(V)

    #df['光模块电流(mA)'].replace('-', '0', inplace=True)
    #print(df)