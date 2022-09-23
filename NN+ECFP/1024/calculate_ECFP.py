# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:48:25 2018

@author: l
"""

from rdkit.Chem import AllChem
from rdkit import Chem
import pandas as pd
import numpy as np

def generator(input_path, output_path):
    data = pd.read_csv(input_path, header = 0).iloc[:, -1]
    ECFP_array = []
    for count, smi in enumerate(data):
        #print(smi)
        if count % 100 == 0:
            print('Calculating descriptors of molecule No.%d' %(count))
        mol = Chem.MolFromSmiles(smi)
        mol2 = Chem.AddHs(mol)
        #AllChem.EmbedMolecule(mol2)#, embed_para)
        #AllChem.MMFFOptimizeMolecule(mol2, maxIters = 1000, mmffVariant='MMFF94s')
        ECFP = AllChem.GetMorganFingerprintAsBitVect(mol2,2,nBits=1024)
        #print(np.asarray(ECFP))
        ECFP_array.append(ECFP)
    ECFP_array = np.asarray(ECFP_array)
    np.save(output_path, ECFP_array)
if __name__ == '__main__':
   print('The program is running!')
   for ls in ['train', 'test', 'validation', 'indep']:
       input_path = ls + '.csv'
       output_path = ls + '_descriptor.npy'
       generator(input_path, output_path)
    



