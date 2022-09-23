# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 08:49:39 2021

@author: Administrator
"""

import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    #descriptor_path = ls + '_descriptor.npy'
    #label_path = ls + '.csv'
    #labels = pd.read_csv('descritors_by_rdkit.csv',header = 0)
    #print(descriptors)
    #labels = pd.read_csv('MP_REMOVED_ERROR.csv',header = 0).iloc[:,-2]
    #print(labels)
    #df = pd.concat([descriptors,labels],axis=1)
    X_train = np.load('train_descriptor.npy')
    print(X_train.shape)
    Y_train = pd.read_csv('train.csv', header = 0).iloc[:,-2]
    print(Y_train.shape)
    #X_train, Y_train = train.iloc[:,:-2], train.iloc[:,-2]
    #print(Y_train)
    #scaler = preprocessing.StandardScaler().fit(X_train)
    
    #X_train = scaler.transform(X_train)
    #print(X_train.shape)
    X_test = np.load('test_descriptor.npy')
    Y_test = pd.read_csv('test.csv', header = 0).iloc[:,-2]
    X_val = np.load('validation_descriptor.npy')
    Y_val = pd.read_csv('validation.csv', header = 0).iloc[:,-2]
    #X_test = scaler.transform(X_test)
    #X_val = scaler.transform(X_val)
    #for x in np.isnan(X_train).any():
        #print(x)
    NN = MLPRegressor(hidden_layer_sizes = (256,128), random_state=1, 
                      max_iter=1, early_stopping = False, verbose = True, 
                      n_iter_no_change=50, warm_start=True)
    mae_train = []
    mae_val = []
    mae_test = []
    r2_train = []
    r2_val = []
    r2_test = []
    rmse_train = []
    rmse_val = []
    rmse_test = []
    
    best_index = float("inf")
    best_mae = float("inf")
    
    for i in range(1000):
        
        NN.fit(X_train, Y_train)
        y_train = NN.predict(X_train)
        y_val = NN.predict(X_val)
        y_test = NN.predict(X_test)
        
        mae_val_iter = mean_absolute_error(Y_val, y_val)
        
        mae_train.append(mean_absolute_error(Y_train, y_train))
        mae_val.append(mae_val_iter)
        mae_test.append(mean_absolute_error(Y_test, y_test))
        
        r2_train.append(r2_score(Y_train, y_train))
        r2_val.append(r2_score(Y_val, y_val))
        r2_test.append(r2_score(Y_test, y_test))
        
        rmse_train.append(math.sqrt(mean_squared_error(Y_train, y_train)))
        rmse_val.append(math.sqrt(mean_squared_error(Y_val, y_val)))
        rmse_test.append(math.sqrt(mean_squared_error(Y_test, y_test)))
        
        if mae_val_iter < best_mae:
            best_index = i
            best_mae = mae_val_iter
            dump(NN, 'MP_ECFP.joblib')
        
        if i-best_index > 50 :
            break
      
    print(mae_val)
    np.save("mae_train", mae_train)
    np.save("mae_val", mae_val)
    np.save("mae_test", mae_test)
    np.save("r2_train", r2_train)
    np.save("r2_val", r2_val)
    np.save("r2_test", r2_test)
    np.save("rmse_train", rmse_train)
    np.save("rmse_val", rmse_val)
    np.save("rmse_test", rmse_test)
    
    
    #dump(scaler,'scaler.joblib')
    #train.to_csv('train.csv')
    #validation.to_csv('validation.csv')
    #test.to_csv('test.csv')
    
def evaluate():
    MODEL = load('MP_ECFP.joblib')
    #Scaler = load('scaler.joblib')
    #indep_data = pd.read_csv('indep_descriptor.csv',header = 0)
    X_indep = np.load('test_descriptor.npy')
    Y_indep = pd.read_csv('test.csv', header = 0).iloc[:,-2]
    #X_indep = Scaler.transform(X_indep)
    y_indep = MODEL.predict(X_indep)
    y_indep = pd.DataFrame(y_indep, columns=['prediction'])
    y_indep.to_csv('test_prediction.csv', header = True, index = False)
    #mae_test = np.load("mae_test.npy")
    #mae_val = np.load("mae_val.npy")
    #mae_train = np.load("mae_train.npy")
    #print(np.min(mae_test), np.min(mae_val), np.min(mae_train))
    #print(MODEL.n_outputs_)
def draw():
    import seaborn as sn
    import matplotlib.pyplot as plt
    sn.set_theme(context='paper', style='darkgrid', palette='deep', 
                 font='arial', font_scale=1.5, color_codes=True, rc={"lines.linewidth": 2})
    mae_train = np.load('mae_train.npy')
    
    mae_test = np.load('mae_test.npy')
    mae_val = np.load('mae_val.npy')
    rmse_train = np.load('rmse_train.npy')
    rmse_test = np.load('rmse_val.npy')
    rmse_val = np.load('rmse_test.npy')
    
    r2_train = np.load('r2_train.npy')
    r2_test = np.load('r2_val.npy')
    r2_val = np.load('r2_test.npy')
    
    index = np.argmin(mae_val)
    print(np.min(mae_val))
    print(mae_train[index], mae_val[index], mae_test[index], rmse_train[index], rmse_val[index], rmse_test[index])
    print(r2_train[index], r2_test[index], r2_val[index])
    #sn.lineplot(x = range(mae_train.shape[0]), y = mae_train)
    #sn.lineplot(x = range(mae_train.shape[0]), y = mae_val)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('ECFP_1024&NN MODEL')
    ax1.plot(range(mae_train.shape[0]), mae_train, 'o--')
    ax1.plot(range(mae_train.shape[0]), mae_val, 'x--')
    #ax1.plot(range(mae_train.shape[0]), mae_test, 'D--')
    ax1.set_ylabel('MAE')
    ax1.set_ylim(10,60)
    ax2.plot(range(rmse_train.shape[0]), rmse_train, 'o--')
    ax2.plot(range(rmse_train.shape[0]), rmse_val, 'x--')
    #ax2.plot(range(rmse_train.shape[0]), rmse_test, 'D--')
    ax2.set_ylabel('RMSE')
    ax2.set_ylim(20,60)
    ax2.set_xlabel('Epoch')
    fig.show()
    
#train()
#evaluate()
draw()