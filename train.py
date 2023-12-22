from dataset import UNSW_NB15
from model import TabMT, OrderedEmbedding

# test

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
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

dataset = UNSW_NB15(data_csv=args.data_csv, dtype_xlsx=args.dtype_xlsx, num_clusters=args.num_clusters)
occs = dataset.get_cluster_centers()
cat_dicts = dataset.get_categorical_dicts()
never_mask = dataset.get_label_idx()
labels = dataset.get_label_column()
num_ft = len(occs)

train_idx, test_idx= train_test_split(np.arange(len(labels)), test_size=1-args.train_size, stratify=labels, shuffle=True, random_state=42)
train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)

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
# model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

class TrainingMetrics():
    def __init__(self, num_ft):
        self.record = [[[], []] for ft in range(num_ft)]
    
    def append(self, predictions, truths, i):
        p = predictions.detach().cpu().tolist()
        t = truths.detach().cpu().tolist()
        
        self.record[i][0].extend(p)
        self.record[i][1].extend(t)
        return None
    
    def compute(self): 
        macroF1 = np.array([f1_score(self.record[i][1], self.record[i][0], average='macro') for i in range(len(self.record)) if len(self.record[i][0]) > 0])
        accuracies = np.array([accuracy_score(self.record[i][1], self.record[i][0]) for i in range(len(self.record)) if len(self.record[i][0]) > 0])
        return macroF1, accuracies

def train(dataloader):
    tracker = TrainingMetrics(num_ft)
    total_loss = total_correct = item_count = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}") 

            batch = batch.to(device)
            optimizer.zero_grad()

            y, mask = model(batch)

            loss = 0
            for ft, y_ft in enumerate(y):
                truth = batch[mask[:, ft], ft]
                pred = y_ft[mask[:, ft]]
                
                pred = pred[truth != -1]
                truth = truth[truth != -1]
                
                if (len(truth) > 0):
                    loss += criterion(pred, truth.long())
                
                    tracker.append(pred.argmax(dim=1), truth, ft)
                    total_correct += sum(pred.argmax(dim=1) == truth).item()
                    item_count += pred.shape[0]
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count, num_feats=len(y))

    macroF1, accuracies = tracker.compute()
        
    return total_loss / item_count, macroF1, accuracies

def validate(dataloader):
    with torch.no_grad():
        tracker = TrainingMetrics(num_ft)
        total_loss = total_correct = item_count = 0
        with tqdm.tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Validation Epoch {epoch}") 
    
                batch = batch.to(device)
                y, mask = model(batch)
    
                loss = 0
                for ft, y_ft in enumerate(y):
                    truth = batch[mask[:, ft], ft]
                    pred = y_ft[mask[:, ft]]
                
                    pred = pred[truth != -1]
                    truth = truth[truth != -1]
                    
                    if (len(truth) > 0):
                        loss += criterion(pred, truth.long())
                
                        tracker.append(pred.argmax(dim=1), truth, ft)
                        total_correct += sum(pred.argmax(dim=1) == truth).item()
                        item_count += pred.shape[0]
        
                total_loss += loss.item()
                
                tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count, num_feats=len(y))
            
        macroF1, accuracies = tracker.compute()
        
        return total_loss / item_count, macroF1, accuracies


if (args.wandb):
    wandb.login()
    wandb.init(project='TabMT', 
               config=vars(args),
               name=args.savename)

save_path = 'saved_models/' + args.savename
personal_best = 1000
for epoch in range(args.epochs):
    model.train(True)
    t_loss, t_macroF1, t_accuracies = train(train_loader)
    print(f't_accuracy: {t_accuracies.mean()}, t_F1: {t_macroF1.mean()}')
    
    model.eval()
    
    test_input = torch.zeros((5, num_ft), dtype=torch.long, device=device) - 1
    print(model.gen_batch(test_input))
    
    v_loss, v_macroF1, v_accuracies = validate(test_loader)
    print(f't_accuracy: {v_accuracies.mean()}, t_F1: {v_macroF1.mean()}')
    
    if (v_loss < personal_best):
        personal_best = v_loss
        torch.save(model.state_dict(), save_path)

    if (args.wandb):
        wandb.log({"train_mean_accuracy": t_accuracies.mean(),
                   "train_accuracies": dict(enumerate(t_accuracies)),
                   "train_loss": t_loss,
                   "validation_mean_accuracy": v_accuracies.mean(),
                   "validation_accuracies": dict(enumerate(v_accuracies)),
                   "validation_loss": v_loss,
                   "train_mean_macroF1": t_macroF1.mean(),
                   "train_macroF1s": dict(enumerate(t_macroF1)),
                   "validation_mean_macroF1": v_macroF1.mean(),
                   "validation_macroF1s": dict(enumerate(v_macroF1)),
                   "epoch": epoch})
    