# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from data_preprocessing import read_files, load_data, load_time_data, load_data_v2, load_xgb_data,load_xgb_data_v2, data_aug, read_files_v3, load_time_data_v2, data_aug_v2
from model import Regression_Model, LSTM_model
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV


filepath = 'data/standard_1/ready_1.csv'
#device = torch.device('cuda:0')
"""
filepath2 = 'data/standard_1/ready_2.csv'
filepath3 = 'data/standard_1/ready_3.csv'
filepath4 = 'data/standard_1/ready_4.csv'
filepath5 = 'data/standard_1/ready_5.csv'
"""
files = 11
batch_size = 64
learning_rate = 0.001
epochs = 300
pretrain = False
patience = 50
method = 'xgb' #xgb, reg or time
use_cls = False

def evaluate(epoch, model, data, loss):
    model = model.to('cpu')
    model.eval()
    total_mse = 0.
    cnt = 0.
    for step, (x, y) in enumerate(data):
        #optimizer.zero_grad()
        outputs = model(x)
        mse = loss(outputs,y)
        total_mse += mse.item()
        cnt += 1.0
    total_mse = total_mse / cnt
    print(f"epoch:{epoch}, eval_loss:{total_mse}")
    return total_mse

def train(model, train_data, val_data, optimizer, loss, method):
    best_val_mse = 1e9
    time = 0
    for e in tqdm(range(epochs)):
        model.train()
        cnt = 0.
        total_loss = 0.
        for step, (x,y) in enumerate(train_data):
            optimizer.zero_grad()
            #print(x.shape)
            outputs = model(x)
            #print(outputs)
            batch_loss = loss(outputs, y)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            cnt += 1.0
        total_loss = total_loss/cnt
        print(f"epoch:{e}, loss:{total_loss}")
        mse = evaluate(e,model,val_data,loss)
        if mse < best_val_mse:
            best_val_mse = mse
            if method == 'reg':
                if use_cls:
                    torch.save(model.state_dict(),f'save_models/save_reg_cls.pt')
                else:
                    torch.save(model.state_dict(), f'save_models/save_reg.pt')
            else:
                if use_cls:
                    torch.save(model.state_dict(), f'save_models/save_time_cls.pt')
                else:
                    torch.save(model.state_dict(), f'save_models/save_time.pt')
            time = 0
        else:
            time += 1
            if time == patience:
                break

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
            outputs = model(x, future_dim)
            #print(outputs)
            batch_loss = loss(outputs, y)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            cnt += 1.0
        total_loss = total_loss/cnt
        print(f"epoch:{e}, loss:{total_loss}")
        #mse = evaluate(e,model,val_data,loss)
        if total_loss < best_mse:
            best_mse = total_loss

            torch.save(model.state_dict(), f'save_models/save_time_v2.pt')
            time = 0
        else:
            time += 1
            if time == patience:
                break

def run_deep_v2(raw_data):
    train_set, dim, future_dim = load_time_data_v2(raw_data, 64)
    # model = Regression_Model(dim, 128, 3)

    model = LSTM_model(dim, 64, 1)
    if pretrain:
        global epochs
        model.load_state_dict(torch.load(f'save_models/save_time_v2.pt'), strict=False)
        epochs = 100
    loss = nn.MSELoss(reduction='mean')
    if use_cls:
        loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    train_time(model, train_set, future_dim, optimizer, loss)

def run_deep_model(raw_data):
    if use_cls:
        train_set, val_set, dim = load_data_v2(raw_data, batch_size)
    else:
        train_set, val_set, dim = load_data(raw_data, batch_size)
    # model = Regression_Model(dim, 128, 3)
    if method == 'reg':
        model = Regression_Model(dim, 128, 3)
        if use_cls:
            model = Regression_Model(dim, 128, 2)
            print(model)
    else:
        model = LSTM_model(dim, 64, 3)
        if use_cls:
            model = LSTM_model(dim, 64, 2)
            print(model)
    if pretrain:
        global epochs
        if use_cls:
            model.load_state_dict(torch.load(f'save_models/save_{method}_cls.pt'), strict=False)
        else:
            model.load_state_dict(torch.load(f'save_models/save_{method}.pt'), strict=False)
        epochs = 100
    loss = nn.MSELoss(reduction='mean')
    if use_cls:
        loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    train(model, train_set, val_set, optimizer, loss, method=method)

def run_xgb_model(raw_data):
    #print(raw_data.shape)
    if use_cls:
        X_train, X_test, y_train, y_test = load_xgb_data_v2(raw_data)
        model = XGBClassifier(n_estimators=750, scale_pos_weight=2, max_depth=6)
    else:
        X_train, X_test, y_train, y_test = load_xgb_data(raw_data)
        model = XGBRegressor(n_estimators = 750, scale_pos_weight = 2, max_depth = 6)
    #model = XGBRegressor(n_estimators = 750, scale_pos_weight = 2, max_depth = 6)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if use_cls:
        print(accuracy_score(y_test,y_pred))
        model.save_model('save_models/save_xgb_cls.json')
    else:
        print(mean_squared_error(y_test, y_pred))
        model.save_model('save_models/save_xgb.json')

def adj_params(X_train, y_train):
    """模型调参"""
    params = {
              # 'n_estimators': [20, 50, 100, 150, 200],
              'n_estimators': [500, 750, 1000, 1250],
              }

    # model_adj = XGBRegressor()

    other_params = {'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 123, 'scale_pos_weight':2, 'max_depth':5}
    model_adj = XGBRegressor(**other_params)

    # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
    optimized_param = GridSearchCV(estimator=model_adj, param_grid=params, scoring='r2', cv=5, verbose=1)
    # 模型训练
    optimized_param.fit(X_train, y_train)

    # 对应参数的k折交叉验证平均得分
    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    for mean, param in zip(means, params):
        print("mean_score: %f,  params: %r" % (mean, param))
    # 最佳模型参数
    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
    # 最佳参数模型得分
    print('最佳模型得分:{0}'.format(optimized_param.best_score_))


if __name__ == "__main__":
    raw_data, _ = read_files_v3(16)
    """
    raw_data = read_files(filepath)
    for i in range(2,files+1):
        file_path = f'data/standard_1/ready_{i}.csv'
        new_raw_data = read_files(file_path)
        #print(np.sum(new_raw_data[:,8] <= -27.0))
        if i == 3:
            new_raw_data = new_raw_data[:, [0]+list(range(2,13))]
        raw_data = np.concatenate([raw_data, new_raw_data], axis=0)
    """
    """
    raw_data_2 = read_files(filepath2)
    #print(raw_data_2.shape)
    raw_data_3 = read_files(filepath3)
    raw_data_3 = raw_data_3[:,[0,2,3,4,5,6,7,8,9,10,11, 12]]
    raw_data_4 = read_files(filepath4)
    #print(raw_data_3.max(axis=0))
    #print(raw_data_3.min(axis=0))
    raw_data = np.concatenate([raw_data_1, raw_data_2], axis=0)
    raw_data = np.concatenate([raw_data, raw_data_3], axis=0)
    raw_data = np.concatenate([raw_data, raw_data_4], axis=0)
    
    """
    #print(raw_data.shape)
    #run_deep_model(raw_data)
    #raw_data = data_aug(raw_data)
    raw_data = data_aug_v2(raw_data)
    #X_train, X_test, y_train, y_test = load_xgb_data(raw_data)
    #adj_params(X_train,y_train)
    #print(raw_data.shape)
    #print(raw_data[:,8])
    """
    if method == 'xgb':
        run_xgb_model(raw_data)
    else:
        run_deep_model(raw_data)
    """
    run_deep_v2(raw_data)


