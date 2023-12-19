from model import TabMT
from dataset import UNSW_NB15, ReverseTokenizer
import torch
import torch.nn as nn
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

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
              num_feat=num_ft,
              never_mask=[-1, -2]).to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('saved_models/' + args.savename))
model.eval()

print('Recovered model!')

x = torch.zeros((args.rows, num_ft), dtype=torch.long, device=device) - 1
x[:, -13:] = 3 

y, conf = model.module.gen_batch(x)
    
print(y)
print(conf)
    
decoder = ReverseTokenizer(cat_dicts, occs, num_ft)
df = decoder.decode(y.detach().cpu())

print(df)

# df.to_csv('synthetics/' + args.output_name, index=False)