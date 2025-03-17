import pickle
from rdkit import Chem
import torch
import random
import numpy as np
import time
import pandas as pd
from torch import nn
from torch.nn import functional as F


def save_to_pickle(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def canonicalize(smi, clear_stereo=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if clear_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def sdf2smiles(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smiles)

def mol_from_smiles(smi):
    smi = canonicalize(smi)
    if smi is None:
        return None
    return Chem.MolFromSmiles(smi)

def sdf2mol(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path)
    mol = next(suppl)
    return mol

def mol_to_smiles(mol):
    smi = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonicalize(smi)

def mols_to_smiles(mols):
    return [mol_to_smiles(m) for m in mols]

def mols_from_smiles(smiles):
    mols = []
    for smi in smiles:
        mol = mol_from_smiles(smi)
        if mol is not None:
            mols.append(mol)
    return mols

def set_random_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32-1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def mask_valid_molecules(smiles):
    valid_mask = []

    for smi in smiles:
        try:
            mol = mol_from_smiles(smi)
            valid_mask.append(mol is not None)
        except Exception:
            valid_mask.append(False)

    return np.array(valid_mask)


def mask_novel_molecules(smiles, data_smiles):
    novel_mask = []

    for smi in smiles:
        novel_mask.append(smi not in data_smiles)

    return np.array(novel_mask)


def mask_unique_molecules(smiles):
    uniques, unique_mask = set(), []

    for smi in smiles:
        unique_mask.append(smi not in uniques)
        uniques.add(smi)

    return np.array(unique_mask)

def score_samples(samples, dataset, calc=True):
    def ratio(mask):
        total = mask.shape[0]
        if total == 0:
            return 0.0
        return mask.sum() / total

    if isinstance(samples, pd.DataFrame):
        smiles = samples.smiles.tolist()
    elif isinstance(samples, list):
        smiles = [s[0] for s in samples]
    data_smiles = dataset.smiles.tolist()

    valid_mask = mask_valid_molecules(smiles)
    novel_mask = mask_novel_molecules(smiles, data_smiles)
    unique_mask = mask_unique_molecules(smiles)

    scores = []
    if calc:
        start = time.time()
        print("Start scoring...")
        validity_score = ratio(valid_mask)
        novelty_score = ratio(novel_mask[valid_mask])
        uniqueness_score = ratio(unique_mask[valid_mask])

        print(f"valid: {validity_score} - "
              f"novel: {novelty_score} - "
              f"unique: {uniqueness_score}")

        scores = [validity_score, novelty_score, uniqueness_score]
        end = time.time() - start
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end))
        print(f'Done. Time elapsed: {elapsed}.')

    return valid_mask * novel_mask * unique_mask, scores

def collate_fn(batch):
    return batch 

def load_checkpoint(checkpoint_path, model):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dicts['model_state_dict'])
    print('\tEpoch {}'.format(state_dicts['epoch']))
    return

def loss_nll(S, log_probs, mask):
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

class Fragloss(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, output, target, mu, sigma):
        output = output.float()
        output = F.log_softmax(output, dim=-1)
        target = target[:,:-1]
        target = target.reshape(-1)
        output = output.view(-1, output.size(-1))
        mask = (target != self.pad).float()
        nb_tokens = int(torch.sum(mask).item())
        output = output[torch.arange(output.size(0)), target] * mask
        CE_loss = -torch.sum(output) / nb_tokens
        KL_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return CE_loss + KL_loss
    
def set_random_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32-1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed