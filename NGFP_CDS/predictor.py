# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:21:52 2020

@author: Administrator
"""

import torch as T
import pandas as pd
from NeuralGraph import preprocessing as prep
from NeuralGraph.dataset import MolData
from torch.utils.data import DataLoader
from descriptor_generator import generator
import numpy as np
from joblib import load, dump


net = T.load('output/GNP_CDS/gcn_MP.pkg')
data_source = 'train'
indep = pd.read_csv('dataset/'+data_source+'.csv')
scaler = load('output/scaler.joblib')

for i in range(7,8): 
    
    start_num = i*10000
    if i == 7:
        #end_num = -1
        cmps = indep.iloc[start_num:, -1]
        cds = indep.iloc[start_num:, 1:-2]
    else :
        end_num = start_num + 10000
        cmps = indep.iloc[start_num:end_num, -1]
        cds = indep.iloc[start_num:end_num, 1:-2]
    
    falke_label = np.zeros((len(cmps)))
    #print(indep.iloc[:,0:-2])
    
    #cds = indep.iloc[start_num:end_num, 1:-2]#generator(cmps)
    indep_x_cds = scaler.transform(cds)
    tesor_smiles = MolData(cmps,falke_label, indep_x_cds)
    predict_loader = DataLoader(tesor_smiles)
    #print(tesor_smiles)
    values, weight = net.predict(predict_loader)
    
    prediction = pd.DataFrame(values, columns = ['Prediction'])
    prediction.to_csv(data_source+'_prediction_GNP_CDS_'+str(i+1)+'.csv', header = True, index=False)
    print(values-273.15)
    dump(weight,data_source+'_weight.joblib')