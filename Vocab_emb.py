import pandas as pd
from gensim.models import Word2Vec 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

SOS_TOKEN = '<SOS>' 
PAD_TOKEN = '<PAD>' 
EOS_TOKEN = '<EOS>' 
TOKENS = [SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]
embed_size = 256
embed_window = 3 

def train_embeddings(data): 
    start_idx = len(TOKENS) 
    modified_sentences = [s.split(" ") for s in data.fragments] 

    w2i = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    w2v = Word2Vec(modified_sentences, vector_size=embed_size, window=embed_window, min_count=1, negative=5, workers=20, epochs=10, sg=1)

    vocab = w2v.wv.key_to_index 
    w2i.update({word: index + start_idx for word, index in vocab.items()}) 

    i2w = {v: k for (k, v) in w2i.items()} 

    tokens = np.random.uniform(-0.05, 0.05, size=(start_idx, embed_size)) 
    embeddings = np.zeros((start_idx + len(vocab), embed_size))
    embeddings[:start_idx, :] = tokens
    for idx, word in enumerate(vocab):
        embeddings[start_idx + idx, :] = w2v.wv[word]

    directory_path = 'F:/Mymodel/data'
    file_name = f'frag_emb_{embed_size}.dat'
    path = f"{directory_path}/{file_name}"
    np.savetxt(path, embeddings, delimiter=",")

    return w2i, i2w


class Vocab: 
    def __init__(self, data): 
        w2i, i2w = train_embeddings(data)
        self.w2i = w2i
        self.i2w = i2w
        self.size = len(self.w2i)
    
    def get_size(self): 
        return self.size
    
    def _translate_integer(self, index): 
        word = self.i2w[index]
        return word

    def _translate_string(self, word): 
        if word in self.w2i:
            return self.w2i[word]
        else:
            return self.w2i[PAD_TOKEN] 

    def get(self, value): 
        if isinstance(value, str):
            return self._translate_string(value)
        elif isinstance(value, int) or isinstance(value, np.integer):
            return self._translate_integer(value)
        raise ValueError('Value type not supported.')
    
    def translate(self, values): 
        res = []
        for v in values:
            if v not in self.TOKEN_IDS: 
                res.append(self.get(v))
            if v == self.EOS: 
                break
        return res

    def append_delimiters(self, sentence):
        return [SOS_TOKEN] + sentence + [EOS_TOKEN]
    
    @property
    def EOS(self): 
        return self.w2i[EOS_TOKEN]

    @property
    def PAD(self):
        return self.w2i[PAD_TOKEN]

    @property
    def SOS(self):
        return self.w2i[SOS_TOKEN]

    @property
    def TOKEN_IDS(self): 
        return [self.SOS, self.EOS, self.PAD]


class DataCollator:
    def __init__(self, vocab):
        self.vocab = vocab

    def merge(self, sequences): 
        sequences = sorted(sequences, key=len, reverse=True) 
        lengths = [len(seq) for seq in sequences]
        padded_seqs = np.full((len(sequences), max(lengths)), self.vocab.PAD)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return torch.LongTensor(padded_seqs), lengths

    def __call__(self, data):
        src_seqs, tgt_seqs = zip(*data)
        src_seqs, src_lengths = self.merge(src_seqs)
        tgt_seqs, tgt_lengths = self.merge(tgt_seqs)
        return src_seqs, tgt_seqs, src_lengths

class FragmentDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('data/clean_data.csv')
        self.data = data.reset_index(drop=True)
        self.size = self.data.shape[0]
        self.vocab = None

    def __getitem__(self, index):
        seq = self.data.fragments[index].split(" ")
        seq = self.vocab.append_delimiters(seq) 
        src = self.vocab.translate(seq[:-1])
        tgt = self.vocab.translate(seq[1:])
        return src, tgt

    def __len__(self): 
        return self.size
    
    def get_loader(self):
        collator = DataCollator(self.vocab)
        loader = DataLoader(dataset=self,
                            collate_fn=collator,
                            batch_size=4,
                            shuffle=True)
        print(f'Data loaded. Size: {self.size}.')
        return loader

    def get_vocab(self):
        self.vocab = Vocab(self.data)
        print(f'Vocab created/loaded. '
              f'Size: {self.vocab.get_size()}. ')

        return self.vocab


