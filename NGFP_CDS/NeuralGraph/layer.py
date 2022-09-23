import torch as T
from torch import nn
from torch.nn import functional as F
import numpy as np
from .util import lookup_neighbors


class GraphConv(nn.Module):
    ''' Hidden Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
    2015). This layer takes a graph as an input. The graph is represented as by
    three tensors.
    - The atoms tensor represents the features of the nodes.
    - The bonds tensor represents the features of the edges.
    - The edges tensor represents the connectivity (which atoms are connected to
        which)
    It returns the convolved features tensor, which is very similar to the atoms
    tensor. Instead of each node being represented by a num_atom_features-sized
    vector, each node now is represented by a convolved feature vector of size
    conv_width.
    '''
    def __init__(self, input_dim, conv_width, max_degree=5):
        super(GraphConv, self).__init__()
        self.conv_width = conv_width
        self.max_degree = max_degree
        self.inner_3D_layers = nn.ModuleList([nn.Linear(input_dim, self.conv_width) for _ in range(max_degree)])
        # for degree in range(max_degree):

    def forward(self, *input, mask=None):
        atoms, bonds, edges = input

        # Create a matrix that stores the degree for each atom
        atom_degrees = (edges != -1).sum(-1, keepdim=True)

        # For each atom, look up the features of it's neighbour
        neighbor_atom_features = lookup_neighbors(atoms, edges, include_self=True)

        # Sum along degree axis to get summed neighbour features
        summed_atom_features = neighbor_atom_features.sum(-2)

        # Sum the edge features for each atom
        summed_bond_features = bonds.sum(-2)

        # Concatenate the summed atom and bond features
        summed_features = T.cat([summed_atom_features, summed_bond_features], dim=-1)
        
        # For each degree we convolve with a different weight matrix
        new_features = None
        for degree in range(self.max_degree):
            atom_masks_this_degree = (atom_degrees == degree).float()
            new_unmasked_features = F.relu(self.inner_3D_layers[degree](summed_features))
            # Do explicit masking because TimeDistributed does not support masking
            new_masked_features = new_unmasked_features * atom_masks_this_degree

            new_features = new_masked_features if degree == 0 else new_features + new_masked_features

        # Finally sum the features of all atoms
        return new_features


class GraphOutput(nn.Module):
    """ Output Convolutional layer in a Neural Graph (as in Duvenaud et. al.,
        2015). This layer takes a graph as an input. The graph is represented as by
        three tensors.
        - The atoms tensor represents the features of the nodes.
        - The bonds tensor represents the features of the edges.
        - The edges tensor represents the connectivity (which atoms are connected to
            which)
        It returns the fingerprint vector for each sample for the given layer.
        According to the original paper, the fingerprint outputs of each hidden layer
        need to be summed in the end to come up with the final fingerprint.
    """

    def __init__(self, input_dim=128, output_dim=128):
        super(GraphOutput, self).__init__()
        self.fp_len = output_dim
        self.inner_3D_layer = nn.Linear(input_dim, self.fp_len)

    def forward(self, atoms, bonds, edges):

        # Create a matrix that stores for each atom, the degree it is, use it
        #   to create a general atom mask (unused atoms are 0 padded)
        # We have to use the edge vector for this, because in theory, a convolution
        #   could lead to a zero vector for an atom that is present in the molecule
        atom_degrees = (edges != -1).sum(-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()

        # Sum the edge features for each atom
        summed_bond_features = bonds.sum(-2)

        # Concatenate the summed atom and bond features
        summed_features = T.cat([atoms, summed_bond_features], dim=-1)

        #Compute fingerprint
        fingerprint_out_unmasked = T.tanh(self.inner_3D_layer(summed_features))

        # Do explicit masking because TimeDistributed does not support masking
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask

        # Sum across all atoms
        final_fp_out = fingerprint_out_masked.sum(dim=-2)
        return final_fp_out


class GraphPool(nn.Module):
    """ Pooling layer in a Neural graph, for each atom, takes the max for each
        feature between the atom and it's neighbours
        # Input shape
            List of Atom and edge tensors of shape:
            `[(samples, max_atoms, atom_features), (samples, max_atoms, max_degrees,
              bond_features), (samples, max_atoms, max_degrees)]`
            where degrees referes to number of neighbours
        # Output shape
            New atom features (of same shape:)
            `(samples, max_atoms, atom_features)`
        """
    def __init__(self):
        super(GraphPool, self).__init__()

    def forward(self, atoms, edges):
        # For each atom, look up the featues of it's neighbour
        neighbor_atom_features = lookup_neighbors(atoms, edges, maskvalue=-np.inf, include_self=True)
        # For each atom, look up the featues of it's neighbour
        max_features = neighbor_atom_features.max(dim=2)[0]
        atom_degrees = (edges != -1).sum(dim=-1, keepdim=True)
        general_atom_mask = (atom_degrees != 0).float()
        return max_features * general_atom_mask
