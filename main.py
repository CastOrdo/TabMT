from dataset import UNSW_NB15
from model import TabMT, OrderedEmbedding

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import tqdm

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

dataset = UNSW_NB15(data_csv='UNSW-NB15_back up/UNSW-NB15_1.csv',
                   dtype_xlsx='UNSW-NB15_back up/NUSW-NB15_features.xlsx',
                   num_clusters=10)

dataloader = DataLoader(dataset, batch_size=40, shuffle=True)

occs = dataset.get_occs()
cat_dicts = dataset.get_categorical_dicts()

model = TabMT(width=64, 
              depth=12, 
              heads=4, 
              occs=occs,
              dropout=0.1,
              dim_feedforward=2000, 
              tu=[0.5 for i in range(len(occs) + len(cat_dicts))], 
              cat_dicts=cat_dicts)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

def train(dataloader):
    total_loss = 0
    item_count = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}") 

            batch = batch.to(device)
            optimizer.zero_grad()

            y, x = model(batch)

            loss = 0
            for i, feat in enumerate(y):
                print(feat)
                print(x[:, i])
                loss += criterion(feat, x[:, i].long())
            
            loss.backward()
            optimizer.step()

            item_count += len(y) * batch.shape[0]
            total_loss += loss.item()
            
            tepoch.set_postfix(loss=total_loss / item_count)
    
    return total_loss / item_count

for epoch in range(20):
    model.train(True)
    t_loss = train(dataloader)
    