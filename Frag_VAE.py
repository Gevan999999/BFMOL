import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class Frag_Encoder(nn.Module):
    def __init__(self, embed_size=256, hidden_size=128, hidden_layers=2, latent_size=256, dropout=0.3, use_gpu=False):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.use_gpu = use_gpu
        self.rnn = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.hidden_layers, dropout=dropout, batch_first=True) 
        self.rnn2mean = nn.Linear(in_features=self.hidden_size * self.hidden_layers, out_features=self.latent_size)
        self.rnn2logv = nn.Linear(in_features=self.hidden_size * self.hidden_layers, out_features=self.latent_size)

    def forward(self, inputs, embeddings, lengths): 
        batch_size = inputs.size(0) 
        state = self.init_state(dim=batch_size) 
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
        _, state = self.rnn(packed, state) 
        state = state.view(batch_size, self.hidden_size * self.hidden_layers) 
        mean = self.rnn2mean(state)
        logv = self.rnn2logv(state)
        std = torch.exp(0.5 * logv)
        z = self.reparameterization_trick(mean, std)
        return z, mean, std  

    def reparameterization_trick(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        latent_z = eps.mul(std).add_(mu)
        return latent_z

    
    def init_state(self, dim):
        state = torch.zeros((self.hidden_layers, dim, self.hidden_size))
        return state.cuda() if self.use_gpu else state

    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, queries, mask=None):
        N, query_len, embed_size = queries.shape
        key_len, value_len = keys.shape[1], values.shape[1]

        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        return self.fc_out(out)


class Frag_Decoder(nn.Module):
    def __init__(self, embed_size, compose_size, hidden_size,
                 hidden_layers, dropout, output_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.latent_to_hidden = nn.Linear(compose_size, hidden_size)
        self.dropout = dropout
        self.num_heads = num_heads

        self.rnn = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size,
                          num_layers=self.hidden_layers, dropout=self.dropout, batch_first=True)
        self.attention = SelfAttention(embed_size=self.hidden_size, heads=self.num_heads)
        self.rnn2out = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, embeddings, state, lengths):
        batch_size = embeddings.size(0)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hidden, state = self.rnn(packed, state)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        attended_hidden = self.attention(hidden, hidden, hidden)
        output = self.rnn2out(attended_hidden)
        return output, state

    def latent_vector_to_hidden(self, latent_z): 
        latent_z = latent_z.repeat(self.hidden_layers, 1, 1)
        hidden = self.latent_to_hidden(latent_z)
        return hidden
