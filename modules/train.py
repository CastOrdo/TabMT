import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules.dataset import stratified_sample

from modules.evaluation import compute_catboost_utility

import wandb
import tqdm
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

g = torch.Generator()
g.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def train(dataloader, model):
    total_loss = total_correct = item_count = 0
    with tqdm.tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}") 
            
            rows, mask = batch
            rows, mask = rows.to(device), mask.to(device)

            optimizer.zero_grad()
            
            y = model(rows, mask)

            loss = 0
            for ft, y_ft in enumerate(y):
                m = (mask[:, ft] == 1) & (rows[:, ft] != -1)
                truth = rows[m, ft]
                pred = y_ft[m]

                if (len(truth) > 0):
                    loss += criterion(pred, truth.long())
                    total_correct += sum(pred.argmax(dim=1) == truth).item()
                    item_count += pred.shape[0]
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            
            tepoch.set_postfix(loss=total_loss / item_count, accuracy=total_correct / item_count)
        
    return total_loss / item_count, total_correct / item_count
    
def fit(model, 
        dataset, 
        num_clusters,  
        train_size,
        target,  
        lr, 
        epochs, 
        batch_size, 
        weight_decay,  
        savename='optuna', 
        save_to_wandb=0):
    
    global optimizer, scheduler, criterion, epoch
    
    frame = dataset.get_frame()
    train_idx = stratified_sample(y=frame[target], lengths=[train_size])[0]
    train_loader = DataLoader(Subset(dataset, train_idx), 
                              batch_size=batch_size, 
                              shuffle=True, 
                              generator=g,
                              worker_init_fn=seed_worker)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = math.ceil(len(train_idx) / batch_size) * epochs
    scheduler = CosineAnnealingLR(optimizer, steps)
    
    model = model.to(device)
    model.train(True)
    
    save_path = 'saved_models/' + savename
    lowest_loss = 10000
    
    for epoch in range(epochs):
        t_loss, t_accuracy = train(train_loader, model)
        
        if t_loss < lowest_loss:
            lowest_loss = t_loss
            torch.save(model.state_dict(), save_path)

        if (save_to_wandb):
            wandb.log({"train_mean_accuracy": t_accuracy,
                       "train_loss": t_loss,
                       "epoch":epoch})
    
    model.load_state_dict(torch.load(save_path))
    return model