import numpy as np
from rdkit import Chem
from . import feature
from tqdm import tqdm
import torch as T
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as rddes
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit.ForceField.rdForceField import ForceField
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.EState import Fingerprinter


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

def calculate_cds(m):
    mol2 = m
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
    d_n_Cno2_2 = len(mol2.GetSubstructMatches(patt5)) - 3 * d_n_Cno2_3
    
    d_n_oc = Chem.Fragments.fr_methoxy(mol2)
    patt7 = Chem.MolFromSmiles('C(C)')
    d_n_c = len(mol2.GetSubstructMatches(patt7))

    
    if (d_n_Cno2_3 != 0) or (d_n_Cno2_2 != 0) :
        d_n_Cno2 = len(mol2.GetSubstructMatches(patt3)) - 3 * d_n_Cno2_3 - 2 * d_n_Cno2_2
    else :
        d_n_Cno2 = len(mol2.GetSubstructMatches(patt3))
    d_1 = rddes.CalcNumHBA(mol2)
    d_2 = rddes.CalcNumHBD(mol2)
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
        #nonenumberlist.append(count)
        print('Sonemthing wrong, please check!')
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
    estate = Estate_calculator(mol2)
    d_mol = d_mol + estate.tolist()
    #print(d_mol)
    return np.asarray(d_mol)
   
def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    """ Padds one axis of an array to a new size
    This is just a wrapper for np.pad, more useful when only padding a single axis
    # Arguments:
        array: the array to pad
        new_size: the new size of the specified axis
        axis: axis along which to pad
        pad_value: pad value,
        pad_right: boolean, pad on the right or left side
    # Returns:
        padded_array: np.array
    """
    add_size = new_size - array.shape[axis]
    pad_width = [(0, 0)] * len(array.shape)

    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)


def tensorise_smiles(smiles, max_degree=5, max_atoms=80):
    """Takes a list of smiles and turns the graphs in tensor representation.
    # Arguments:
        smiles: a list (or iterable) of smiles representations
        max_atoms: the maximum number of atoms per molecule (to which all
            molecules will be padded), use `None` for auto
        max_degree: max_atoms: the maximum number of neigbour per atom that each
            molecule can have (to which all molecules will be padded), use `None`
            for auto
        **NOTE**: It is not recommended to set max_degree to `None`/auto when
            using `NeuralGraph` layers. Max_degree determines the number of
            trainable parameters and is essentially a hyperparameter.
            While models can be rebuilt using different `max_atoms`, they cannot
            be rebuild for different values of `max_degree`, as the architecture
            will be different.
            For organic molecules `max_degree=5` is a good value (Duvenaud et. al, 2015)
    # Returns:
        atoms: np.array, An atom feature np.array of size `(molecules, max_atoms, atom_features)`
        bonds: np.array, A bonds np.array of size `(molecules, max_atoms, max_neighbours)`
        edges: np.array, A connectivity array of size `(molecules, max_atoms, max_neighbours, bond_features)`
    """

    # import sizes
    n = len(smiles)
    n_atom_features = feature.num_atom_features()
    n_bond_features = feature.num_bond_features()

    # preallocate atom tensor with 0's and bond tensor with -1 (because of 0 index)
    # If max_degree or max_atoms is set to None (auto), initialise dim as small
    #   as possible (1)
    atom_tensor = np.zeros((n, max_atoms or 1, n_atom_features))
    bond_tensor = np.zeros((n, max_atoms or 1, max_degree or 1, n_bond_features))
    edge_tensor = -np.ones((n, max_atoms or 1, max_degree or 1), dtype=int)
    #CDS_tensor = np.zeros((n, 104))

    for mol_ix, s in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(s)
        #mol = Chem.AddHs(mol)
        
        try:
            AllChem.EmbedMolecule(mol)
            #AllChem.UFFOptimizeMolecule(mol)
        except:
            #nonenumberlist.append(count)
            print('Molcule %d error, please check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' %(mol_ix))
            continue
        #AllChem.ComputeGasteigerCharges(mol)
        #CDS_tensor[mol_ix] = calculate_cds(mol)
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        

        # If max_atoms is exceeded, resize if max_atoms=None (auto), else raise
        if len(atoms) > atom_tensor.shape[1]:
            atom_tensor = padaxis(atom_tensor, len(atoms), axis=1)
            bond_tensor = padaxis(bond_tensor, len(atoms), axis=1)
            edge_tensor = padaxis(edge_tensor, len(atoms), axis=1, pad_value=-1)
        rdkit_ix_lookup = {}

        for atom_ix, atom in enumerate(atoms):
            # write atom features
            #print(atom.GetDoubleProp("_GasteigerCharge"))
            atom_tensor[mol_ix, atom_ix, : n_atom_features] = feature.atom_features(atom)
            # store entry in idx
            rdkit_ix_lookup[atom.GetIdx()] = atom_ix

        # preallocate array with neighbor lists (indexed by atom)
        connectivity_mat = [[] for _ in atoms]

        for bond in bonds:
            # lookup atom ids
            a1_ix = rdkit_ix_lookup[bond.GetBeginAtom().GetIdx()]
            a2_ix = rdkit_ix_lookup[bond.GetEndAtom().GetIdx()]

            # lookup how many neighbours are encoded yet
            a1_neigh = len(connectivity_mat[a1_ix])
            a2_neigh = len(connectivity_mat[a2_ix])

            # If max_degree is exceeded, resize if max_degree=None (auto), else raise
            new_degree = max(a1_neigh, a2_neigh) + 1
            if new_degree > bond_tensor.shape[2]:
                assert max_degree is None, 'too many neighours ({0}) in molecule: {1}'.format(new_degree, s)
                bond_tensor = padaxis(bond_tensor, new_degree, axis=2)
                edge_tensor = padaxis(edge_tensor, new_degree, axis=2, pad_value=-1)

            # store bond features
            bond_features = np.array(feature.bond_features(bond), dtype=int)
            bond_tensor[mol_ix, a1_ix, a1_neigh, :] = bond_features
            bond_tensor[mol_ix, a2_ix, a2_neigh, :] = bond_features

            # add to connectivity matrix
            connectivity_mat[a1_ix].append(a2_ix)
            connectivity_mat[a2_ix].append(a1_ix)

        # store connectivity matrix
        for a1_ix, neighbours in enumerate(connectivity_mat):
            degree = len(neighbours)
            edge_tensor[mol_ix, a1_ix, : degree] = neighbours
    #print(CDS_tensor)
    return T.from_numpy(atom_tensor).float(), \
           T.from_numpy(bond_tensor).float(), \
           T.from_numpy(edge_tensor).long(),
           #T.from_numpy(CDS_tensor).float()
