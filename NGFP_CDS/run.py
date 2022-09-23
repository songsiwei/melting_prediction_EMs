#from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from NeuralGraph.dataset import MolData
from NeuralGraph.model import QSAR
import pandas as pd
import numpy as np
from torch import nn
import torch as T
import random
from sklearn import preprocessing
from joblib import dump

def setup_seed(seed):
     T.manual_seed(seed)
     T.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     T.backends.cudnn.deterministic = True
     
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)
        
def main(target_id, include_CDS):
    setup_seed(666)
    train = pd.read_csv('dataset/train.csv').iloc[0:100,:]
    validation = pd.read_csv('dataset/validation.csv').iloc[0:100,:]
    test = pd.read_csv('dataset/test.csv').iloc[0:100,:]
    X_cds_train = train.iloc[:,1:-2]
    scaler = preprocessing.StandardScaler().fit(X_cds_train)
    dump(scaler,'output/scaler.joblib')
    X_cds_train = scaler.transform(X_cds_train)
    
    X_cds_valid = validation.iloc[:,1:-2]
    X_cds_valid = scaler.transform(X_cds_valid)
    X_cds_test = test.iloc[:,1:-2]
    X_cds_test = scaler.transform(X_cds_test)

    X_trained = train.iloc[:, -1]
    Y_trained = train.iloc[:, -2]
    X_valid = validation.iloc[:, -1]
    Y_valid = validation.iloc[:, -2]
    

    test_set = MolData(np.asarray(test.iloc[:, -1]), np.asarray(test.iloc[:, -2]), X_cds_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    
    if include_CDS:
        out = 'output/GNP_CDS/gcn%s' % ('_' + target_id if target_id else '')
    else:
        out = 'output/GNP/gcn%s' % ('_' + target_id if target_id else '')
    
    
    net = QSAR(in_dim = 37, hid_dim = 256, out_dim=1, if_CDS = include_CDS)
    net.apply(weights_init)
    #train_set, valid_set = data.iloc[trained], data.iloc[valided]
    train_set = MolData(np.asarray(X_trained), np.asarray(Y_trained), X_cds_train)
    valid_set = MolData(np.asarray(X_valid), np.asarray(Y_valid), X_cds_valid)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    #print(len(train_loader))
    net, tr, va = net.fit(train_loader, valid_loader, epochs=N_EPOCH, path='%s' % (out))
    
    
    with T.no_grad():
        print_log = open(out+'_log.txt','w')
        print('Evaluation of MAE, RMSE and R2 in validation Set: %f, %f and %f' % net.evaluate(valid_loader),file = print_log)
        print('Evaluation of MAE, RMSE and R2 in test Set: %f, %f and %f' % net.evaluate(test_loader),file = print_log)
        print_log.close()
        valid_prediction, valid_fp = net.predict(valid_loader)
        test_prediction, test_fp = net.predict(test_loader)
        tr_va = np.hstack((tr, va))
        tr_va = pd.DataFrame(tr_va)
        tr_va.to_csv(out + '_tr&va_' + '.csv',header = ['train_mae', 'train_rmse', 'train_R2', 'valid_mae', 'valid_rmse', 'valid_R2',])
    #print(valid_prediction)
    data_score, test_score = pd.DataFrame(), pd.DataFrame()
    
    data_score['SMILES'] = X_valid
    data_score['LABEL'] = Y_valid
    data_score['PREDICTION'] = valid_prediction
    
    test_score['SMILES'] = test.iloc[:,-1]
    test_score['LABEL'] = test.iloc[:,-2]
    test_score['PREDICTION'] = test_prediction
    
    data_score.to_csv(out + '_valid.csv')
    test_score.to_csv(out + '_test.csv')
    dump(valid_fp, out + '_valid_fp.joblib')
    dump(test_fp, out + '_test_fp.joblib')


if __name__ == '__main__':
    BATCH_SIZE = 10
    N_EPOCH = 1000
    main('MP', include_CDS = True)
