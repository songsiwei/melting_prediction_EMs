import torch as T
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from . import feature
from . import preprocessing as prep


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, smiles, labels, cds_des):
        self.max_atom = 80
        self.max_degree = 5
        self.atoms, self.bonds, self.edges = self._featurize(smiles)
        #print(self.cds)
        self.cds = T.from_numpy(cds_des).float()
        #print(self.cds.shape)
        self.label = T.from_numpy(labels).float()

    def _featurize(self, smiles):
        return prep.tensorise_smiles(smiles, max_atoms=self.max_atom, max_degree=self.max_degree)

    def __getitem__(self, i):
        return self.atoms[i], self.bonds[i], self.edges[i], self.cds[i], self.label[i]

    def split(self, batch_size):
        return

    def __len__(self):
        return len(self.label)

