import torch
import torch.nn as nn
import torch.nn.functional as F
import Bio.PDB
import numpy as np
from Bio.SeqUtils import seq1

class AtomEmbLayer(nn.Module):
    def __init__(self, atom_emb_in=7, atom_emb_h=256):
        super(AtomEmbLayer, self).__init__()
        
        self.atom_norm = nn.InstanceNorm1d(atom_emb_in)
        self.atom_linear = nn.Linear(atom_emb_in, atom_emb_h, bias=False)
        self.atom_activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.atom_norm2 = nn.InstanceNorm1d(atom_emb_h)
        self.atom_linear2 = nn.Linear(atom_emb_h, atom_emb_h, bias=False)

    def forward(self, atom_feat):
        atom_emb = self.atom_norm(atom_feat) 
        atom_emb = self.atom_linear(atom_emb) 
        atom_emb = torch.mean(atom_emb, 1) 

        atom_emb = self.atom_activation(atom_emb) 

        atom_emb = self.atom_norm2(atom_emb.unsqueeze(1)).squeeze() 
        atom_emb = self.atom_linear2(atom_emb)
        atom_emb = self.atom_activation(atom_emb)

        return atom_emb 
    
class NodeEmbLayer(nn.Module):
    def __init__(self, node_emb_in=27, node_emb_h=256):
        super(NodeEmbLayer, self).__init__()

        self.res_norm = nn.InstanceNorm1d(node_emb_in)
        self.res_linear = nn.Linear(node_emb_in, node_emb_h, bias=False)
        self.res_activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.node_linear = nn.Linear(2 * node_emb_h, node_emb_h, bias=False)
        self.node_norm = nn.InstanceNorm1d(node_emb_h)
        self.node_activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, res_feat, atom_emb):
        res_emb = self.res_norm(res_feat) 
        res_emb = self.res_linear(res_emb)  
        res_emb = self.res_activation(res_emb)
        node_emb = torch.cat((res_emb, atom_emb), dim=1) 

        node_emb = self.node_linear(node_emb)
        node_emb = self.node_norm(node_emb)      
        node_emb = self.node_activation(node_emb)  

        return node_emb 
    
class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings=16, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, E_idx):
        Num_nodes = E_idx.size(0)
        ii = torch.arange(Num_nodes, dtype=torch.float32).view((-1, 1)) 
        d = (E_idx.float() - ii).unsqueeze(-1)

        frequency = torch.exp(torch.arange(0, self.num_embeddings, 2, dtype=torch.float32) * -(np.log(10000.0) / self.num_embeddings))
        angles = d * frequency.view((1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class EdgeEmbLayer(nn.Module):
    def __init__(self, edge_emb_in=17, node_emb_h=256):
        super(EdgeEmbLayer, self).__init__()

        self.edge_norm = nn.InstanceNorm1d(edge_emb_in)
        self.edge_linear = nn.Linear(edge_emb_in, node_emb_h, bias=False)
        self.edge_activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, edge_feat):
        edge_emb = self.edge_linear(edge_feat)  
        edge_emb = self.edge_norm(edge_emb) 
        edge_emb = self.edge_activation(edge_emb)
        return edge_emb 
    

class PocketFeatures(nn.Module):
    def __init__(self, top_k=30):
        super(PocketFeatures, self).__init__()
        self.top_k = top_k

        self.atom_embs = AtomEmbLayer()
        self.node_embs = NodeEmbLayer()
        self.pos_embs = PositionalEncodings()
        self.edge_embs = EdgeEmbLayer()

    def get_atom_feat(self, pdb_file):
        pdb_parser = Bio.PDB.PDBParser(QUIET = True)
        structure = pdb_parser.get_structure('pocket', pdb_file)
        residue_list = list(structure.get_residues())
        atom_embs = []

        for residue in residue_list:
            atom_pos, onehot = [], []
            atom_dict = {'N':0, 'CA':1, 'C':2, 'O':3}

            for atom_name in atom_dict.keys():  
                if atom_name in residue:
                    atom_pos.append(residue[atom_name].coord)
                    onehot.append(np.eye(len(atom_dict))[atom_dict[atom_name]])

            CA_pose = residue['CA'].coord
            atom_embs.append(np.concatenate([onehot, np.array(atom_pos) - CA_pose], axis=1).astype(np.float32))

        embedding = np.zeros((len(residue_list), 4, 7))
        for i, emb in enumerate(atom_embs):
            if emb.size > 0:
                embedding[i, :emb.shape[0], :] = emb

        atom_feat = torch.from_numpy(embedding)
        return atom_feat.float()

    def get_pdb_sequence(self, pdb_file):
        parser = Bio.PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("pocket", pdb_file)
        sequence = []
        for residue in structure.get_residues():
            if residue.get_id()[0] == " ":
                resname = seq1(residue.get_resname()) if residue.has_id('CA') else "X"
                sequence.append(resname)
        
        return "".join(sequence).upper()

    def get_seq_onehot(self, seq):
        AA_TYPES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
        aa_index = {aa: idx for idx, aa in enumerate(AA_TYPES)}

        seq_len = len(seq)
        onehot_matrix = np.zeros((seq_len, len(AA_TYPES)), dtype=int)
        for i, res in enumerate(seq):
            index = aa_index.get(res, aa_index["X"])
            onehot_matrix[i, index] = 1
        
        return onehot_matrix 

    def get_coord_matrix(self, pdb_file):
        atom_coords = {}
        atom_types = ['N', 'CA', 'C', 'O']
        res_order = []

        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    res_id = line[22:26].strip()
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    if res_id not in atom_coords:
                        atom_coords[res_id] = {atom: [0, 0, 0] for atom in atom_types}
                        res_order.append(res_id)
                    
                    if atom_name in atom_types:
                        atom_coords[res_id][atom_name] = [x, y, z]

        num_residues = len(res_order)
        coord_matrix = np.zeros((num_residues, len(atom_types), 3))

        for i, res_id in enumerate(res_order):
            for j, atom in enumerate(atom_types):
                coord_matrix[i, j, :] = atom_coords[res_id].get(atom, [0, 0, 0])

        return torch.tensor(coord_matrix, dtype=torch.float32) 


    def get_dihedrals(self, coord_matrix, eps=1e-7):
        coord_matrix = coord_matrix[:,:3,:].reshape(3*coord_matrix.shape[0], 3)  
        dC = coord_matrix[1:,:] - coord_matrix[:-1,:]  
        U = F.normalize(dC, dim=-1) 
        
        u_2 = U[:-2,:]  
        u_1 = U[1:-1,:]  
        u_0 = U[2:,:]  
        
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1) 
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)  

        cosD = (n_2 * n_1).sum(-1)  
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)  

        D = F.pad(D, (1,2), 'constant', 0)  
        D = D.view(int(D.size(0)/3), 3)  

        dihedrals = torch.cat((torch.cos(D), torch.sin(D)), 1)  
        return dihedrals 

    def dist(self, coord_matrix, eps=1E-6):
        num_residues = coord_matrix.shape[0]
        if num_residues < self.top_k:
            mask = torch.ones(self.top_k, dtype=torch.float32)
            padding = torch.zeros((self.top_k - num_residues, 4, 3), dtype=torch.float32)
            X = torch.cat((coord_matrix, padding), dim=0)
            mask[num_residues:] = 0

            residue_centers = torch.mean(X, dim=1)  
            dX = torch.unsqueeze(residue_centers, 1) - torch.unsqueeze(residue_centers, 0)  
            D = torch.sqrt(torch.sum(dX**2, dim=-1) + eps)  

            mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 0)  
            D = D * mask_2D  #
            D_max, _ = torch.max(D, dim=-1, keepdim=True)  
            D_adjust = D + (1. - mask_2D) * D_max  
        else:
            residue_centers = torch.mean(coord_matrix, dim=1)  
            dX = torch.unsqueeze(residue_centers, 1) - torch.unsqueeze(residue_centers, 0)  
            D = torch.sqrt(torch.sum(dX**2, dim=-1) + eps)  
        
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False)
        D_neighbors = torch.unsqueeze(D_neighbors, -1)
        
        return D_neighbors, E_idx

    def forward(self, pdb_file):
        atom_feat = self.get_atom_feat(pdb_file)
        atom_embs = self.atom_embs(atom_feat)
        sequence = self.get_pdb_sequence(pdb_file)
        onehot_matrix = self.get_seq_onehot(sequence)
        coord_matrix = self.get_coord_matrix(pdb_file)
        dihedrals = self.get_dihedrals(coord_matrix)
        node_feat = np.concatenate((onehot_matrix, dihedrals), axis=1)
        res_feat = torch.tensor(node_feat, dtype=torch.float32)
        node_embs = self.node_embs(res_feat, atom_embs)

        D_neighbors, E_idx = self.dist(coord_matrix)
        E_positional = self.pos_embs(E_idx)
        edge_feat = torch.cat((E_positional, D_neighbors), -1)
        edge_embs = self.edge_embs(edge_feat)

        return node_embs, edge_embs, E_idx

def cat_neighbors_nodes(h_V, h_E, E_idx):
    if E_idx.size(1) > h_V.size(0):
        E_idx = E_idx[:h_V.size(0), :h_V.size(0)]
        h_E = h_E[:h_V.size(0), :h_V.size(0)]

    neighbors_flat = E_idx.reshape(-1)
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, h_V.size(1))
    neighbor_features = torch.gather(h_V, 0, neighbors_flat)
    neighbor_features = neighbor_features.view(E_idx.shape[0], E_idx.shape[1], -1)
    h_nn = torch.cat([h_E, neighbor_features], -1)
    return h_nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W1 = nn.Linear(num_hidden, num_ff, bias=True)
        self.W2 = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        return self.W2(F.relu(self.W1(h_V)))
    
class MPNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scales=30):
        super(MPNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scales
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.InstanceNorm1d(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm(h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm(h_V + self.dropout(dh))
        h_V = torch.mean(h_V, dim=0)
        return h_V
        
class Space_Encoder(nn.Module):
    def __init__(self, top_k=30, input_dim=256,hidden_dim=256,z_dim=256,dropout=0.1):
        super(Space_Encoder, self).__init__()
        self.input_size = input_dim
        self.dropout = dropout
        self.features = PocketFeatures(top_k=top_k)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer = MPNN(hidden_dim, hidden_dim*2, dropout=dropout)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logv = nn.Linear(hidden_dim, z_dim)

    def forward(self, pdb_batch):
        featembs = []
        for pdb in pdb_batch:
            nodes, edges, E_idx = self.features(pdb)
            h_V = self.W_v(nodes)
            h_E = self.W_e(edges)
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = self.layer(h_V, h_EV)
            featembs.append(h_V)
            
        V = torch.stack(featembs, dim=0)
        mu = self.mean(V)
        sigma = self.logv(V)
        z = self.reparameterization_trick(mu, sigma)
        return z, mu, sigma

    def reparameterization_trick(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_z = eps.mul(std).add_(mu)
        return latent_z

