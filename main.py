from modules.dataset import UNSW_NB15, stratified_sample
from modules.model import TabMT
from modules.train import fit
from modules.evaluation import compute_catboost_utility
import numpy as np
import torch
import random
import wandb

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--num_clusters', type=int, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)


parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--dropout', type=float, required=True)
parser.add_argument('--weight_decay', type=float, required=True)

parser.add_argument('--savename', type=str, required=True)
parser.add_argument('--save_to_wandb', type=int, required=True)
args = parser.parse_args()

train_size = 50000
utility_train_size = utility_test_size = 50000

data_csv = ['data/UNSW_NB_15_1_withCVSS_V2.csv', 
            'data/UNSW_NB_15_2_withCVSS_V2.csv',
            'data/UNSW_NB_15_3_withCVSS_V2.csv',
            'data/UNSW_NB_15_4_withCVSS_V2.csv']
dtype_xlsx = 'data/NUSW-NB15_features.xlsx'
dropped_columns = ['label']
labels = ['attack_cat', 'cvss']

dataset = UNSW_NB15(data_csv=data_csv, 
                    dtype_xlsx=dtype_xlsx, 
                    dropped_columns=dropped_columns,
                    labels=labels,
                    n_clusters=args.num_clusters)

meta = dataset.get_meta()

model = TabMT(width=args.width, 
              depth=args.depth, 
              heads=args.heads, 
              encoder_list=meta['encoder_list'],
              dropout=args.dropout)

if args.save_to_wandb:
    wandb.login()
    wandb.init(project='TabMT', config=vars(args), name=args.savename)

model = fit(model=model, 
            dataset=dataset, 
            target='cvss',
            num_clusters=args.num_clusters, 
            train_size=train_size, 
            lr=args.lr, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            weight_decay=args.weight_decay,
            device=device,
            save_to_wandb=args.save_to_wandb, 
            savename=args.savename)

print('Starting Evaluation!')

model.eval()

frame = meta['raw_frame'].dropna()
means, stds = compute_catboost_utility(model=model,
                                       frame=frame, 
                                       train_size=utility_train_size, 
                                       test_size=utility_test_size, 
                                       target='cvss', 
                                       labels=labels, 
                                       dtypes=meta['dtypes'], 
                                       device=device, 
                                       num_exp=2, 
                                       num_trials=2)

results = {"mean_accuracy_diff": means[0],
           "mean_macroF1_diff":means[1],
           "mean_weightedF1_diff":means[2],
           "mean_macroGM_diff":means[3],
           "mean_weightedGM_diff":means[4],
           "std_accuracy_diff": stds[0],
           "std_macroF1_diff":stds[1],
           "std_weightedF1_diff":stds[2],
           "std_macroGM_diff":stds[3],
           "std_weightedGM_diff":stds[4]}

print(results)

if args.save_to_wandb:
    wandb.log(results)