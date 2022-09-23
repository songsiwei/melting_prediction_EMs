# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:48:25 2018

@author: l
"""
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
import xlwt,xlrd
import rdkit.Chem.rdMolDescriptors as rddes
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import ETKDGv2
from rdkit.ForceField.rdForceField import ForceField
from rdkit.Chem import rdForceFieldHelpers
#from openbabel import pybel
#from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT
import numpy as np
import pandas as pd
from rdkit.Chem.EState import Fingerprinter
#from rdkit.Chem.rdmolfiles import SmilesMolSupplier as smilestomolecule
#from rdkit.Chem.rdmolfiles import SDWriter as sdwriter

descriptors_name = ['nHbondA','nHbondD','nNH2','nAHC','nACC',
                    'nHC','nRbond','nR','nNNO2','nONO2',
                    'nNO2','nC(NO2)3','nC(NO2)2','nC(NO2)','MinPartialCharge',
                    'MaxPartialCharge','MOLvolume','nH','nC','nN',
                    'nO','PBF','TPSA','ob','total energy',
                    'molecular weight','PMI3','nOCH3','nCH3',
                    'Eccentricity','PMI2','PMI1','NPR1','NPR2']#'molar refractivity'
listnone_name = ['no. of nonemolecule']
def Estate_calculator(m):
    exestate1, exestate2 = Fingerprinter.FingerprintMol(m)
    estate1 = np.squeeze(exestate1)
    estate2 = np.squeeze(exestate2)
    
    s1 = slice(6,38,1)
    s2 = slice(53,54,1)
    s3 = slice(69,70,1)
    s4 = slice(74,75,1)
    selected_estate_1 = np.append(np.append(np.append(estate1[s1],estate1[s2]),estate1[s3]),estate1[s4])
    selected_estate_2 = np.append(np.append(np.append(estate2[s1],estate2[s2]),estate2[s3]),estate2[s4])
    return np.append(selected_estate_1,selected_estate_2)
    
def generator(input_smis):
    #if 'dataset' in input_filename:
       # dataset = pd.read_excel(input_filename,header = None)
    #elif 'prediction' in input_filename:
       # dataset = pd.read_table(input_filename,header = None)
    smiles = input_smis
    #print(smiles)
    E_state_len = len(Estate_calculator(Chem.MolFromSmiles('CC')))
    total_len = len(descriptors_name) + E_state_len
    list_all = np.zeros((len(smiles),total_len))
    #list_all.append(descriptors_name)
    #count = 0

    nonenumberlist = []

    for count, smi in enumerate(smiles):
        
       # print(smi)
        mol = Chem.MolFromSmiles(smi)
        if (mol is None) == True :
            nonenumberlist.append(count)
            continue
        
        print('Calculating descriptors of molecule No.%d' %(count))
        
        #Chem.AddHs(mol)
        mol2 = Chem.AddHs(mol)
        try:
            #embed_para = ETKDGv2()
            AllChem.EmbedMolecule(mol2)#, embed_para)
            AllChem.UFFOptimizeMolecule(mol2, maxIters = 1000, ignoreInterfragInteractions = True)
            AllChem.MMFFOptimizeMolecule(mol2, maxIters = 1000, mmffVariant='MMFF94s', ignoreInterfragInteractions = True)
            
            
          #  print(iter_flag)
        except:
            nonenumberlist.append(count)
            print('Somemthing wrong, please check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            continue
        
        d_mol = []
        nH = 0
        nC = 0
        nN = 0
        nO = 0
        #add hydrogen and cleanup the 3d structure by MMFF forcefield
        #print(labelofopt)
        for atom in mol2.GetAtoms():
            if atom.GetAtomicNum() == 1:
                nH = nH + 1
            if atom.GetAtomicNum() == 6:
                nC = nC + 1
            if atom.GetAtomicNum() == 7:
                nN = nN + 1
            if atom.GetAtomicNum() == 8:
                nO = nO + 1
        mol_wt = rddes.CalcExactMolWt(mol2)
        d_ob = 16*(nO - 0.5 * nH - 2 * nC)/mol_wt
        
        patt1 = Chem.MolFromSmiles('N(N(=O)=O)')
        d_n_nno2 = len(mol2.GetSubstructMatches(patt1))
        patt2 = Chem.MolFromSmiles('O(N(=O)=O)')
        d_n_ono2 = len(mol2.GetSubstructMatches(patt2))
        patt3 = Chem.MolFromSmiles('C(N(=O)=O)')
        d_n_no2 = len(mol2.GetSubstructMatches(patt3))
        patt4 = Chem.MolFromSmiles('C(N(=O)=O)(N(=O)=O)(N(=O)=O)')
        d_n_Cno2_3 = len(mol2.GetSubstructMatches(patt4))
        patt5 = Chem.MolFromSmiles('C(N(=O)=O)(N(=O)=O)')
        d_n_Cno2_2 = len(mol.GetSubstructMatches(patt5)) - 3 * d_n_Cno2_3
        
        d_n_oc = Chem.Fragments.fr_methoxy(mol2)
        patt7 = Chem.MolFromSmiles('C(C)')
        d_n_c = len(mol2.GetSubstructMatches(patt7))

        
        if (d_n_Cno2_3 != 0) or (d_n_Cno2_2 != 0) :
            d_n_Cno2 = len(mol2.GetSubstructMatches(patt3)) - 3 * d_n_Cno2_3 - 2 * d_n_Cno2_2
        else :
            d_n_Cno2 = len(mol2.GetSubstructMatches(patt3))
        d_1 = rddes.CalcNumHBA(mol2)
        d_2 = rddes.CalcNumHBD(mol2)-d_n_no2
       # d_3 = Descriptors3D.SpherocityIndex(mol2)
      #  d_4 = Chem.Fragments.fr_nitro(mol)
        d_5 = Chem.Fragments.fr_NH2(mol2)
        
       # d_6 = Chem.Fragments.fr_oxazole(mol)
        #d_7 = Chem.Fragments.fr_ester(mol)
       # d_8 = Chem.Fragments.fr_azide(mol)
        d_9 = rddes.CalcNumAromaticHeterocycles(mol2)
        d_10 = rddes.CalcNumAromaticCarbocycles(mol2)
        d_11 = rddes.CalcNumHeterocycles(mol2)
        d_12 = rddes.CalcNumRotatableBonds(mol2)
       # d_13 = rddes.CalcNumLipinskiHBA(mol)
      #  d_14 = rddes.CalcNumLipinskiHBD(mol)
        d_15 = rddes.CalcNumRings(mol2)
        d_16 = Descriptors.MinPartialCharge(mol2, force=True)
        d_17 = Descriptors.MaxPartialCharge(mol2, force=True)
        d_18 = AllChem.ComputeMolVolume(mol2, confId=-1, gridSpacing=0.2, boxMargin=2.0)
        #d_19 = rddes.CalcNumBridgeheadAtoms(mol2, atoms = ['N'])
      #  d_20 = BalabanJ(mol2)
      #  d_21 = BertzCT(mol2)
      #  d_22 = rddes.CalcWHIM(mol2)
        
        d_23 = rddes.CalcPBF(mol2)
        d_24 = Descriptors.TPSA(mol2)
        d_25 = Descriptors.MolWt(mol2)
        d_26 = Descriptors3D.PMI3(mol2)
        #mymol = pybel.readstring("smi", Chem.MolToSmiles(mol2))#.make3D(forcefield='UFF',steps=100)
        #mymol = Chem.MolToMolBlock(mol2)calcdesc(descnames=['MR']
        #d_27 = pybel.Molecule(mymol).calcdesc(descnames=['MR'])
       # print(d_27)
        d_28 = Descriptors3D.Eccentricity(mol2)
        d_29 = Descriptors3D.PMI2(mol2)
        d_30 = Descriptors3D.PMI3(mol2)
        d_31 = Descriptors3D.NPR1(mol2)
        d_32 = Descriptors3D.NPR2(mol2)
       # print(d_23)
        try:
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol2)#, rdForceFieldHelpers.UFFGetMoleculeProperties(mol)
            d_energy = ForceField.CalcEnergy(ff)
        except:
            nonenumberlist.append(count)
            print('Sonemthing wrong, please check!')
            continue
        d_mol.append(d_1)
        d_mol.append(d_2)
      #  d_mol.append(d_3)
        #d_mol.append(d_4)
        d_mol.append(d_5)
        #d_mol.append(d_6)
        #d_mol.append(d_7)
       # d_mol.append(d_8)
        d_mol.append(d_9)
        d_mol.append(d_10)
        d_mol.append(d_11)
        d_mol.append(d_12)
       # d_mol.append(d_13)
       # d_mol.append(d_14)
        d_mol.append(d_15)
        d_mol.append(d_n_nno2)
        d_mol.append(d_n_ono2)
        d_mol.append(d_n_no2)
        d_mol.append(d_n_Cno2_3)
        d_mol.append(d_n_Cno2_2)
        d_mol.append(d_n_Cno2)
        d_mol.append(d_16)
        d_mol.append(d_17)
        d_mol.append(d_18)
        d_mol.append(nH)
        d_mol.append(nC)
        d_mol.append(nN)
        d_mol.append(nO)
       # d_mol.append(d_19)
      #  d_mol.append(d_20)
      #  d_mol.append(d_21)
       # d_mol.append(d_22)
        d_mol.append(d_23)
        d_mol.append(d_24)
        d_mol.append(d_ob)
        d_mol.append(d_energy)
        d_mol.append(d_25)
        d_mol.append(d_26)
        d_mol.append(d_n_oc)
        d_mol.append(d_n_c)
       # d_mol.append(d_27['MR'])
        d_mol.append(d_28)
        d_mol.append(d_29)
        d_mol.append(d_30)
        d_mol.append(d_31)
        d_mol.append(d_32)
       # print(d_mol)
        estate = Estate_calculator(mol)
        #print(estate)
        d_mol = d_mol + estate.tolist()
        #print(d_mol)
       # print(d_mol_new)
        
        list_all[count] = d_mol
        
    #data_write(output_filename, list_all)
    #print(list_all)
    #len_estate = len(list_all[1])
    #composite_descriptors = descriptors_name + ['ESTATE_'+ str(i) for i in range(E_state_len)]
    #df_descriptors = pd.DataFrame(list_all,columns=composite_descriptors)
    #df_descriptors.to_excel(output_filename, index=False)
    #none_mol =pd.DataFrame(nonenumberlist,columns=listnone_name)
    #none_mol.to_excel(input_filename.strip('\.') + '_nonenumberlist.xlsx', index=False)
    return list_all

if __name__ == '__main__':
   print('程序自身在运行')
   generator('dataset/CLASSIFY/MOL.xlsx.','dataset/CLASSIFY/descritors_by_rdkit.xlsx')
    



