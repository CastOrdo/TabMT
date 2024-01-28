from modules.dataset import decode_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from tqdm import tqdm

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
    def __init__(self, classes, embedding_dim):
        super(SimpleEmbedding, self).__init__()
        num_embeddings = len(classes)
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
    def __init__(self, width, depth, heads, encoder_list, dropout=0, tu=None):
        super(TabMT, self).__init__()
        self.width = width
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.encoder_list = encoder_list
        self.num_ft = len(encoder_list)
        self.tu = tu if tu != None else [1 for i in range(self.num_ft)]

        self.Embeddings, self.LinearLayers = nn.ModuleList(), nn.ModuleList()
        for idx, encoder in enumerate(encoder_list):
            continuous = encoder.type_ == 'continuous'
            embedding = OrderedEmbedding(encoder.classes_, self.width) if continuous else SimpleEmbedding(encoder.classes_, self.width)
            linear = DynamicLinear(self.tu[idx])

            self.Embeddings.append(embedding)
            self.LinearLayers.append(linear)
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.width, 
                                                        nhead=self.heads,
                                                        dropout=self.dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.depth)

        self.mask_vec = nn.Parameter(standard_norm.sample((1, self.width)))
        self.positional_encoding = nn.Parameter(position_norm.sample((self.num_ft, self.width)))
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total Trainable Parameters in this Model: {pytorch_total_params}')
    
    def embed(self, x, mask):        
        out = self.mask_vec.repeat(x.shape[0], x.shape[1], 1)
        
        for ft in range(self.num_ft):
            col_mask = (mask[:, ft] == 0) & (x[:, ft] != -1)
            out[col_mask, ft] = self.Embeddings[ft](x[col_mask, ft])
        return out

    def linear(self, x):
        return [self.LinearLayers[ft](x[:, ft], self.Embeddings[ft].weight()) for ft in range(self.num_ft)]

    def forward(self, x, mask):        
        y = self.embed(x, mask)
        y = y + self.positional_encoding
        y = self.encoder(y)
        y = self.linear(y)
        return y
    
    def gen(self, x, batch_size):
        for row in tqdm(torch.split(x, batch_size, dim=0), desc='Generating Data'):
            for i in torch.randperm(self.num_ft):
                mask = torch.zeros(row.size(), device=x.device)
                y = self.embed(row, mask)

                y = y + self.positional_encoding
                y = self.encoder(y)
                y = self.LinearLayers[i](y[:, i], self.Embeddings[i].weight())

                missing = (row[:, i] == -1)
                logits = F.softmax(y[missing], dim=1)
                predictions = torch.multinomial(logits, num_samples=1)
                row[missing, i] = torch.squeeze(predictions)
        return x

    def generate_data(self, label_vectors, label_idx, num_frames, device):
        for i, ft in enumerate(label_idx):
            enc = self.encoder_list[ft]
            label_vectors[:, i] = enc.transform(label_vectors[:, i])

        rows, columns = len(label_vectors), len(self.encoder_list)
        gen_in = torch.ones((rows, columns), dtype=int) * -1
        gen_in[:, label_idx] = torch.from_numpy(label_vectors)
    
        synthetic_frames = []
        for i in range(num_frames):
            gen_in_hat = torch.clone(gen_in)
            gen_in_hat = gen_in_hat.to(device)
            
            synthetic = self.gen(gen_in_hat, batch_size=512)
            synthetic = decode_output(synthetic, self.encoder_list)
            
            synthetic_frames.append(synthetic)
        
        return synthetic_frames
    