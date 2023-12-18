from dataset import UNSW_NB15
from model import TabMT, OrderedEmbedding

# test

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
import numpy as np

import wandb
import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_csv', type=str, nargs='+', required=True)
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
occs = dataset.get_cluster_centers()
cat_dicts = dataset.get_categorical_dicts()
num_ft = len(occs)

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
              cat_dicts=cat_dicts,
              num_feat=num_ft).to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

class WeightedF1():
    def __init__(self, num_ft):
        self.record = [[[], []] for ft in range(num_ft)]
    
    def append(self, predictions, truths, i):
        self.record[i][0].extend(predictions.detach().cpu().tolist())
        self.record[i][1].extend(truths.cpu().tolist())
        return None
    
    def compute(self):
        weightedF1 = np.array([f1_score(self.record[i][1], self.record[i][0], average='weighted') for i in range(len(self.record)) if len(self.record[i][0]) > 0])
        
        mean_weightedF1 = weightedF1.mean()
        return weightedF1, mean_weightedF1

def train(dataloader):
    f1 = WeightedF1(num_ft)
    total_loss = total_correct = item_count = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}") 

            batch = batch.to(device)
            optimizer.zero_grad()

            y, i = model(batch)

            loss = 0
            for ft_idx in i:
                y_col = model.module.fast_linear(y, ft_idx)
                
                loss += criterion(y_col, batch[:, ft_idx].long())

                f1.append(y_col.argmax(dim=1), batch[:, ft_idx], ft_idx)
                total_correct += sum(y_col.argmax(dim=1) == batch[:, ft_idx]).item()
                item_count += y_col.shape[0]
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count, num_feats=len(y))

    weightedF1, mean_weightedF1 = f1.compute()
        
    return total_loss / item_count, total_correct / item_count, weightedF1, mean_weightedF1

def validate(dataloader):
    with torch.no_grad():
        f1 = WeightedF1(num_ft)
        total_loss = total_correct = item_count = 0
        with tqdm.tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Validation Epoch {epoch}") 
    
                batch = batch.to(device)
                y, i = model(batch)
    
                loss = 0
                for ft_idx in i:
                    y_col = model.fast_linear(y, ft_idx)
                    
                    loss += criterion(y_col, batch[:, ft_idx].long())
    
                    f1.append(y_col.argmax(dim=1), batch[:, ft_idx], ft_idx)
                    total_correct += sum(y_col.argmax(dim=1) == batch[:, ft_idx]).item()
                    item_count += y_col.shape[0]
        
                total_loss += loss.item()
                
                tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count, num_feats=len(y))
            
        weightedF1, mean_weightedF1 = f1.compute()    
        
        return total_loss / item_count, total_correct / item_count, weightedF1, mean_weightedF1

if (args.wandb):
    wandb.login()
    wandb.init(project='TabMT', 
               config=vars(args),
               name=args.savename)

save_path = 'saved_models/' + args.savename
personal_best = 1000
for epoch in range(args.epochs):
    model.train(True)
    t_loss, t_acc, t_weightedF1, t_mean_weightedF1 = train(train_loader)
    
    model.eval()
    v_loss, v_acc, v_weightedF1, v_mean_weightedF1 = validate(test_loader)
    
    if (v_loss < personal_best):
        personal_best = v_loss
        torch.save(model.state_dict(), save_path)

    if (args.wandb):
        wandb.log({"train_accuracy": t_acc, 
                   "train_loss": t_loss,
                   "validation_accuracy": v_acc,
                   "validation_loss": v_loss,
                   "train_MWF1": t_mean_weightedF1,
                   "validation_MWF1": v_mean_weightedF1,
                   "epoch": epoch})
    