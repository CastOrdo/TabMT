import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

standard_norm = Normal(0, 0.05)
position_norm = Normal(0, 0.01)

class OrderedEmbedding(nn.Module):
    def __init__(self, occ, width):
        super(OrderedEmbedding, self).__init__()
        self.r = torch.tensor((occ - min(occ)) / (max(occ) - min(occ)), dtype=torch.float).view(-1,1)
        self.E = torch.zeros((len(occ), width), dtype=torch.float)
        self.l, self.h = standard_norm.sample([width]), standard_norm.sample([width])

        self.r, self.E, self.l, self.h = nn.Parameter(self.r), nn.Parameter(self.E), nn.Parameter(self.l), nn.Parameter(self.h)
        
    def weight(self):
        return self.E + self.r * self.l + (1 - self.r) * self.h
    
    def forward(self, idx):
        matrix = self.weight()
        return matrix[idx]

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
    def __init__(self, width, depth, heads, dropout, dim_feedforward, tu, occs, cat_dicts):
        super(TabMT, self).__init__()
        self.width = width
        self.depth = depth
        self.heads = heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.tu = tu
    
        z, i = 0, 0
        self.embeddings = nn.ModuleList()
        self.dynamic_linears = nn.ModuleList()
        for idx in range(len(occs) + len(cat_dicts)):
            if (idx in cat_dicts):
                emb = nn.Embedding(len(cat_dicts[idx]), self.width)
                emb.weight.data = standard_norm.sample((len(cat_dicts[idx]), width))
                i += 1
            else:
                emb = OrderedEmbedding(occs[z], self.width)
                z += 1
            lin = DynamicLinear(self.tu[idx])
            self.embeddings.append(emb)
            self.dynamic_linears.append(lin)
            
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.width, 
                                                        nhead=self.heads,
                                                        dim_feedforward=self.dim_feedforward,
                                                        dropout=self.dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.depth)

        self.mask_vec = nn.Parameter(standard_norm.sample((1, self.width)))
        self.mask_vec.requires_grad = False

        self.positional_encoding = position_norm.sample((len(occs) + len(cat_dicts), self.width))
        self.positional_encoding = nn.Parameter(self.positional_encoding)

    def embed(self, x, mask):
        out = self.mask_vec.repeat(x.shape[0], x.shape[1], 1)
        for ft in range(x.shape[1]):
            col_mask = (mask[ft] == 0) & (x[:, ft] != -1)
            out[col_mask, ft] = self.embeddings[ft](x[col_mask, ft])
        return out

    def linear(self, x, mask):
        out = []
        for idx in range(x.shape[1]):
            if (mask[idx] == 1):
                if isinstance(self.embeddings[idx], OrderedEmbedding):
                    E = self.embeddings[idx].weight()
                else:
                    E = self.embeddings[idx].weight.data
                out.append(self.dynamic_linears[idx](x[:, idx], E))
        return out

    def forward(self, x):
        mask = torch.rand(x.shape[1]).round().int()
        y = self.embed(x, mask)
        y = y + self.positional_encoding
        y = self.encoder(y)
        y = self.linear(y, mask)
        return y, x[:, mask == 1]

    def gen_batch(self, rows):
        with torch.no_grad():
            batch = torch.empty((rows, len(self.embeddings)))
            mask = torch.ones(len(self.embeddings)).int()
            
            for i in torch.randperm(len(self.embeddings)):
                y = self.embed(batch, mask)
                y = y + self.positional_encoding
                y = self.encoder(y)
                y = self.linear(y, mask)

                batch[:, i] = y[i]
                mask[i] = 0
        return batch