import torch
import torch.nn as nn
import numpy as np
from Space_VAE import Space_Encoder
from Frag_VAE import Frag_Encoder, Frag_Decoder
from Vocab_emb import FragmentDataset

dataset = FragmentDataset()
vocab = dataset.get_vocab()

class Space2Frag(nn.Module):
    def __init__(self, use_gpu=False):
        super(Space2Frag, self).__init__()
        self.space_encoder = Space_Encoder()
        self.use_gpu = use_gpu
        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)
        self.frag_encoder = Frag_Encoder(embed_size=256, hidden_size=128, hidden_layers=2, latent_size=256, dropout=0.3, use_gpu = self.use_gpu)
        self.latent2rnn = nn.Linear(in_features=256, out_features=256)
        self.frag_decoder = Frag_Decoder(embed_size=256, compose_size=256, hidden_size=128, 
                                      hidden_layers=2, dropout=0.3, output_size=vocab.get_size(), 
                                      num_heads=8)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, lengths, pdb_batch):
        space_z, space_mu, space_sigma = self.space_encoder(pdb_batch)
        embeddings = self.embedder(src)
        frag_z, mu, sigma = self.frag_encoder(src, embeddings, lengths)
        compose_z =space_z + frag_z
        state = self.latent2rnn(compose_z)
        state = state.view(2, state.size(0), 128) 
        output, state = self.frag_decoder(embeddings, state, lengths)
        return output, mu, sigma
    
    def load_embeddings(self):
        path = 'data/frag_emb_256.dat'
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()

    def sample(self, pdb_batch, max_length=8, temperature=1.0):
        batch_size = len(pdb_batch)
        S = torch.zeros((batch_size, max_length), dtype=torch.long)
        start_token_idx = vocab.get('<SOS>')  
        end_token_idx = vocab.get('<EOS>')    
        with torch.no_grad():
            space_z, _, _ = self.space_encoder(pdb_batch)
            state = None
            S[:, 0] = start_token_idx  
            for i in range(max_length - 1):
                embeddings = self.embedder(S[:, :i+1])  
                if i == 0:
                    frag_z, _, _ = self.frag_encoder(S[:, :1], embeddings, torch.ones(batch_size).long())
                    compose_z = space_z + frag_z
                    state = self.latent2rnn(compose_z).view(2, batch_size, 128)
                output, state = self.frag_decoder(embeddings, state, torch.tensor([i+1] * batch_size, dtype=torch.long))
                
                current_probs = torch.softmax(output[:, i], dim=-1)
                adjusted_probs = torch.pow(current_probs, 1 / temperature)
                next_token = torch.multinomial(adjusted_probs, 1).squeeze(1)
                S[:, i+1] = next_token
                if (next_token == end_token_idx).all():
                    break
        S = S[:, 1:]
        return S


