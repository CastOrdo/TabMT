from dataset import UNSW_NB15
from model import TabMT, OrderedEmbedding

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import wandb
import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_csv', type=str, required=True)
parser.add_argument('--dtype_xlsx', type=str, required=True)

parser.add_argument('--width', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--dim_feedforward', type=int, required=True)

parser.add_argument('--dropout', type=float, required=True)
parser.add_argument('--num_clusters', type=int, required=True)
parser.add_argument('--tu', type=float, required=True)

parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)

parser.add_argument('--savename', type=str, required=True)
parser.add_argument('--wandb', type=int, required=True)

parser.add_argument('--train_size', type=float, required=True)

args = parser.parse_args()

dataset = UNSW_NB15(data_csv=args.data_csv,
                   dtype_xlsx=args.dtype_xlsx,
                   num_clusters=args.num_clusters)
occs = dataset.get_occs()
cat_dicts = dataset.get_categorical_dicts()

train_size = int(args.train_size * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader, test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

model = TabMT(width=args.width, 
              depth=args.depth, 
              heads=args.heads, 
              occs=occs,
              dropout=args.dropout,
              dim_feedforward=args.dim_feedforward, 
              tu=[args.tu for i in range(len(occs) + len(cat_dicts))], 
              cat_dicts=cat_dicts).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

def train(dataloader):
    total_loss = 0
    total_correct = 0
    item_count = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}") 

            batch = batch.to(device)
            optimizer.zero_grad()

            y, x = model(batch)

            loss = 0
            for i, feat in enumerate(y):
                loss += criterion(feat, x[:, i].long())
                total_correct += sum(feat.argmax(dim=1) == x[:, i]).item()
                item_count += feat.shape[0]
                
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count, num_feats=len(y))
    
    return total_loss / item_count, total_correct / item_count

def validate(dataloader):
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        item_count = 0
        with tqdm.tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Validation Epoch {epoch}") 
    
                batch = batch.to(device)
                y, x = model(batch)
    
                loss = 0
                for i, feat in enumerate(y):
                    loss += criterion(feat, x[:, i].long())
                    total_correct += sum(feat.argmax(dim=1) == x[:, i]).item()
                    item_count += feat.shape[0]
    
                item_count += len(y) * batch.shape[0]
                total_loss += loss.item()
                
                tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count, num_feats=len(y))
        
        return total_loss / item_count, total_correct / item_count

if (args.wandb):
    wandb.login()
    wandb.init(project='TabMT', 
               config=vars(args),
               name=args.savename)

save_path = 'saved_models/' + args.savename
personal_best = 0
for epoch in range(args.epochs):
    model.train(True)
    t_loss, t_acc = train(train_loader)
    model.eval()
    v_conf, v_loss = validate(test_loader)
    
    if (v_loss < personal_best):
        personal_best = v_loss
        torch.save(model.state_dict(), save_path)

    if (args.wandb):
        wandb.log({"train_accuracy": t_acc, 
                   "train_loss": t_loss,
                  "validation_accuracy": v_acc,
                  "validation_loss": v_loss,
                  "epoch": epoch})
    