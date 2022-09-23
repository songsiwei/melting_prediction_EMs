import torch as T
from torch import nn
from torch.nn import functional as F
#from torch.nn import MultiheadAttention as MA
from .layer import GraphConv, GraphPool, GraphOutput
import numpy as np
from torch import optim
import time
from .util import dev
import math
from sklearn.metrics import r2_score
#from .GATlayer import GraphAttentionLayer


class QSAR(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, alpha=0.01, nheads = 4, if_CDS = True):
        super(QSAR, self).__init__()
        self.nheads = nheads
        self.if_CDS = if_CDS
        self.gcn1 = GraphConv(input_dim= in_dim, conv_width=64)
        self.gcn2 = GraphConv(input_dim= 70, conv_width=hid_dim)
        self.gop = GraphOutput(input_dim= hid_dim+6, output_dim=128)
        #self.bn1 = nn.BatchNorm2d(1)
        #self.bn2 = nn.BatchNorm1d(1)
        self.pool = GraphPool()
        #self.att_func = MA(256, self.nheads, batch_first=True)#nn.Sequential(*[nn.Linear(128, 1), nn.Sigmoid()])  batch_first=True
        #atten_out_dim = int(256/nheads)
        #self.atten_layer = nn.Linear(self.input_size, self.input_size)
        #self.atten_layer = nn.Sequential(*[nn.Linear(256, self.nheads), nn.Softmax()])
        #self.atten_layer_list = nn.Sequential(*[nn.Linear(256,64), nn.Linear(64,1), nn.Sigmoid()])
        if self.if_CDS:
            linear_in_dim = 128 + 104
        else:
            linear_in_dim = 128
        self.fc1 = nn.Linear(linear_in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)
        #self.fc_atten = nn.Linear(1, 1)
        self.to(dev)

    def forward(self, atoms, bonds, edges, cds_des):

        atoms = self.gcn1(atoms, bonds, edges)
        atoms = self.pool(atoms, edges)
        #atoms = self.pool(atoms, edges)
        atoms = self.gcn2(atoms, bonds, edges)

        atoms = self.pool(atoms, edges)
        
        
        #query = out_atoms.unsqueeze(dim=-2)
        #print(key_value.shape)
        #weighted_atoms, atten_weight_collect = self.att_func(atoms, atoms, atoms)
        
        fp = self.gop(atoms, bonds, edges)
        
        if self.if_CDS :
            fp = T.cat((fp, cds_des), -1)
        #fp = T.cat((fp, cds_des), -1)

        fp1 = F.relu(self.fc1(fp))
        fp2 = F.relu(self.fc2(fp1))
        out = self.fc3(fp2)
        
        


        return out, fp

    def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=50, lr=3e-4):
        criterion = nn.L1Loss(reduction='sum')
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)
        best_loss = np.inf
        last_saving = 0
         
        record_tr = []
        record_va = []
        
        for epoch in range(epochs):
            t0 = time.time()
            sample = 0
            loss_total = 0
            for Ab, Bb, Eb, CDSb, yb in loader_train:
                Ab, Bb, Eb, CDSb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), CDSb.to(dev), yb.to(dev)
                #print(CDSb)
                sample += Ab.shape[0]
                optimizer.zero_grad()
                y_, gcn_fp = self.forward(Ab, Bb, Eb, CDSb)
               # ix = yb == yb
               # yb, y_ = yb[ix], y_[ix]
                y_ = T.squeeze(y_)
                #print(y_)
                loss = criterion(y_, yb)
                loss_total += loss.item()
                #print(loss)
                loss.backward()
                optimizer.step()
            train_mae = loss_total/sample
            with T.no_grad():
                loss_train_mae, loss_train_rmse, R2_train = self.evaluate(loader_train)
                loss_valid_mae, loss_valid_rmse, R2_valid = self.evaluate(loader_valid)
                print('[Epoch:%d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                    epoch+1, epochs, time.time() - t0, train_mae, loss_valid_mae))
                if loss_valid_mae < best_loss:
                    T.save(self, path + '.pkg')
                    #T.save(weight_, path + '.weight')
                    print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (
                        best_loss, loss_valid_mae, path + '.pkg'))
                    best_loss = loss_valid_mae
                    last_saving = epoch
                else:
                    print('[Performance] loss_valid is not improved.')
                record_tr.append([loss_train_mae, loss_train_rmse, R2_train])
                record_va.append([loss_valid_mae, loss_valid_rmse, R2_valid])
                if early_stop is not None and epoch - last_saving > early_stop: break
            print('Learning rate for this epoch is %f ' % scheduler.get_last_lr()[0])
            scheduler.step()
        return T.load(path + '.pkg'), record_tr, record_va

    def evaluate(self, loader):
        loss_mae = 0
        loss_mse = 0
        criterion_mae = nn.L1Loss(reduction='sum')
        criterion_mse = nn.MSELoss(reduction='sum')
        sample = 0
        yls = np.asarray([])
        y_ls = np.asarray([])
        for Ab, Bb, Eb, CDSb, yb in loader:
            Ab, Bb, Eb, CDSb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), CDSb.to(dev), yb.to(dev)
            sample += Ab.shape[0]
            y_, gcn_fp = self.forward(Ab, Bb, Eb, CDSb)
            #ix = yb == yb
            #yb, y_ = yb[ix], y_[ix]
            y_ = T.squeeze(y_)
            yls = np.concatenate((yls, yb))
            y_ls = np.concatenate((y_ls, y_))
            loss_mae += criterion_mae(y_, yb).item()
            loss_mse += criterion_mse(y_, yb).item()
        yls = np.asarray(yls).reshape((-1))
        #print(yls)
        y_ls = np.asarray(y_ls).reshape((-1))
        R2 = r2_score(yls, y_ls)
        #print(sample)
        return loss_mae/sample, math.sqrt(loss_mse/sample), R2

    def predict(self, loader):
        score = []
        gcn_fp_ls = []
        for Ab, Bb, Eb, CDSb, yb in loader:
            Ab, Bb, Eb, CDSb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), CDSb.to(dev), yb.to(dev)
            y_, gcn_fp = self.forward(Ab, Bb, Eb, CDSb)
            score.append(y_.data.cpu())
            gcn_fp_ls.append(gcn_fp)
        score = T.cat(score, dim=0).numpy()
        gcn_fp_all = T.cat(gcn_fp_ls, dim=0).detach().numpy()
        return score, gcn_fp_all
