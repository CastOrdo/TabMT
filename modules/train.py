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

g = torch.Generator().manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        save_to_wandb, 
        savename):
    
    global optimizer, scheduler, criterion, epoch
    
    frame = dataset.get_frame()
    train_idx = stratified_sample(y=frame[target], lengths=[train_size])[0]
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = math.ceil(len(train_idx) / batch_size) * epochs
    scheduler = CosineAnnealingLR(optimizer, steps)
    
    save_path = 'saved_models/' + savename
    lowest_loss = 10000
    
    model = model.to(device)
    model.train(True)
    for epoch in range(epochs):
        t_loss, t_accuracy = train(train_loader, model)
        
        model.eval()
        results = compute_catboost_utility(model=model, 
                                           frame=dataset.get_frame(), 
                                           target_name='cvss', 
                                           names=dataset.names, 
                                           dtypes=dataset.dtypes, 
                                           encoder_list=dataset.encoder_list, 
                                           label_idx=dataset.label_idx, 
                                           train_size=100000, 
                                           test_size=100000)
        print(f'{results[1]}')
        
        if t_loss < lowest_loss:
            lowest_loss = t_loss
            torch.save(model.state_dict(), save_path)

        if (save_to_wandb):
            wandb.log({"train_mean_accuracy": t_accuracy,
                       "train_loss": t_loss,
                       "epoch":epoch})
    
    model.load_state_dict(torch.load(save_path))
    return model