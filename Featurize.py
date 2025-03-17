from Fragmentation import *
from Vocab_emb import *

dataset = FragmentDataset()
vocab = dataset.get_vocab()

def merge(sequences):
    sequences = sorted(sequences, key=len, reverse=True) 
    lengths = [len(seq) for seq in sequences]
    padded_seqs = np.full((len(sequences), max(lengths)), vocab.PAD)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    
    return torch.LongTensor(padded_seqs), lengths

def frag_featurize(batch):
    batch_src = []
    batch_tgt = []
    batch_pdb = []
    batch_lengths = []
    
    for data in batch:
        sdf = data['sdf']
        try:
            smiles = sdf2smiles(sdf)
            if smiles is None:
                continue  
        except Exception as e:
            continue  

        mol = mol_from_smiles(smiles)
        fseq = []
        frags = fragment_iterative(mol)
        length = len(frags)  
        if length == 0:
            continue

        batch_lengths.append(length)
        for frag in frags:
            sm = mol_to_smiles(frag) 
            fseq.append(sm)

        seq = vocab.append_delimiters(fseq)
        src = vocab.translate(seq[:-1])
        tgt = vocab.translate(seq[1:])
        batch_src.append(src)
        batch_tgt.append(tgt)

        pdb = data['pdb']
        batch_pdb.append(pdb)
    
    padded_src, src_lengths = merge(batch_src)
    padded_tgt, tgt_lengths = merge(batch_tgt)

    return padded_src, batch_lengths, padded_tgt, batch_pdb
