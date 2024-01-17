from model import TabMT
from dataset import UNSW_NB15, ReverseTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()

parser.add_argument('--rows', type=int, required=True)

parser.add_argument('--width', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--dim_feedforward', type=int, required=True)

parser.add_argument('--dropout', type=float, required=True)
parser.add_argument('--num_clusters', type=int, required=True)
parser.add_argument('--tu', type=float, required=True)

parser.add_argument('--savename', type=str, required=True)
parser.add_argument('--output_name', type=str, required=True)

args = parser.parse_args()

cat_dicts = pickle.load(open("processed_data/cat_dicts.pkl", "rb"))
occs = pickle.load(open("processed_data/clstr_cntrs.pkl", "rb"))
num_ft = len(occs)

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
model.load_state_dict(torch.load('saved_models/' + args.savename))
model.eval()

print('Recovered model!')
decoder = ReverseTokenizer(cat_dicts, occs, num_ft)

frame = pd.DataFrame()
for r in tqdm(range(args.rows), desc='Generating Data'):
    x = torch.ones((1, num_ft), dtype=torch.long, device=device) * -1
    x[:, [45, 46]] = 0

    y = model.gen_batch(x)
    df = decoder.decode(y.detach().cpu())
    
    frame = pd.concat([frame, df])
    
print(frame)