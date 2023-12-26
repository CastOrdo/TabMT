import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import datetime as dt

standard_norm = Normal(0, 0.05)
position_norm = Normal(0, 0.01)

class OrderedEmbedding(nn.Module):
    def __init__(self, occ, width):
        super(OrderedEmbedding, self).__init__()
        occ = occ.astype('float')
        self.r = torch.tensor((occ - min(occ)) / (max(occ) - min(occ)), dtype=torch.float).view(-1,1)
        self.E = torch.zeros((len(occ), width), dtype=torch.float)
        self.l, self.h = standard_norm.sample([width]), standard_norm.sample([width])

        self.r, self.E, self.l, self.h = nn.Parameter(self.r), nn.Parameter(self.E), nn.Parameter(self.l), nn.Parameter(self.h)
        
    def weight(self):
        return self.E + self.r * self.l + (1 - self.r) * self.h
    
    def forward(self, idx):
        matrix = self.weight()
        return matrix[idx]

class SimpleEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(SimpleEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data = standard_norm.sample((num_embeddings, embedding_dim))

    def weight(self):
        return self.embedding.weight.data

    def forward(self, idx):
        return self.embedding(idx)

class DynamicLinear(nn.Module):
    def __init__(self, tu):
        super(DynamicLinear, self).__init__()
        self.tu = tu
        self.tl = nn.Parameter(standard_norm.sample([1]))
        self.bias = nn.Parameter(standard_norm.sample([1]))
    
    def forward(self, x, E):
        logits = (torch.matmul(x, E.transpose(1, 0)) + self.bias) / (self.tl * self.tu)
        return logits

class TabMT(nn.Module):
    def __init__(self, width, depth, heads, dropout, dim_feedforward, tu, occs, cat_dicts, num_feat):
        super(TabMT, self).__init__()
        self.width = width
        self.depth = depth
        self.heads = heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.tu = tu

        self.Embeddings, self.LinearLayers = nn.ModuleList(), nn.ModuleList()
        for idx in range(num_feat):
            embedding = SimpleEmbedding(len(cat_dicts[idx]), self.width) if (cat_dicts[idx] != None) else OrderedEmbedding(occs[idx], self.width)
            linear = DynamicLinear(self.tu[idx])

            self.Embeddings.append(embedding)
            self.LinearLayers.append(linear)
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.width, 
                                                        nhead=self.heads,
                                                        dim_feedforward=self.dim_feedforward,
                                                        dropout=self.dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.depth)

        self.mask_vec = nn.Parameter(standard_norm.sample((1, self.width)))
        self.positional_encoding = nn.Parameter(position_norm.sample((num_feat, self.width)))
    
    def embed(self, x, mask):        
        out = self.mask_vec.repeat(x.shape[0], x.shape[1], 1)
        
        for ft in range(x.shape[1]):
            col_mask = (mask[:, ft] == 0) & (x[:, ft] != -1)
            out[col_mask, ft] = self.Embeddings[ft](x[col_mask, ft])
        return out

    def linear(self, x):        
        return [self.LinearLayers[ft](x[:, ft], self.Embeddings[ft].weight()) for ft in range(x.shape[1])]

    def forward(self, x, mask):        
        y = self.embed(x, mask)
        y = y + self.positional_encoding
        y = self.encoder(y)
        y = self.linear(y)
        return y

    def gen_batch(self, x):
        for i in torch.randperm(x.shape[1]):
            y = self.embed(x, torch.zeros(x.size(), device=x.device))
            
            y = y + self.positional_encoding
            y = self.encoder(y)
            y = self.LinearLayers[i](y[:, i], self.Embeddings[i].weight())
            
            x[x[:, i] == -1, i] = y[x[:, i] == -1].argmax(dim=1)        
        return x