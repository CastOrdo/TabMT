from modules.evaluation import compute_catboost_utility
from modules.dataset import UNSW_NB15
from modules.model import TabMT
import torch
import numpy as np
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--num_clusters', type=int, required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--heads', type=int, required=True)
parser.add_argument('--savename', type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_csv = ['data/UNSW_NB_15_1_withCVSS_V2.csv', 
            'data/UNSW_NB_15_2_withCVSS_V2.csv',
            'data/UNSW_NB_15_3_withCVSS_V2.csv',
            'data/UNSW_NB_15_4_withCVSS_V2.csv']
dtype_xlsx = 'data/NUSW-NB15_features.xlsx'
dropped_columns = ['label', 'dsport', 'sport']
labels = ['cvss', 'attack_cat']

dataset = UNSW_NB15(data_csv=data_csv, 
                    dtype_xlsx=dtype_xlsx, 
                    dropped_columns=dropped_columns,
                    labels=labels,
                    n_clusters=args.num_clusters)

encoder_list = dataset.get_encoder_list()
model = TabMT(width=args.width, depth=args.depth, heads=args.heads, 
              encoder_list=encoder_list)

save_path = f'saved_models/{args.savename}'
model.load_state_dict(torch.load(save_path))
model.to(device)

model.eval()

train_size = test_size = 50000
num_exp, num_trials = 5, 5

means, stds = compute_catboost_utility(model=model, 
                                       frame=dataset.get_frame(), 
                                       target_name='cvss', 
                                       names=dataset.names, 
                                       dtypes=dataset.dtypes, 
                                       encoder_list=encoder_list, 
                                       label_idx=dataset.label_idx, 
                                       train_size=train_size, 
                                       test_size=test_size, 
                                       num_exp=num_exp, 
                                       num_trials=num_trials)

print(f'AVG: {means}')
print(f'STD: {stds}')