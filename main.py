from modules.dataset import UNSW_NB15
from modules.model import TabMT
from modules.train import fit
from modules.evaluation import compute_catboost_utility
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--train_size', type=int, required=True)

parser.add_argument('--utility_train_size', type=int, required=True)
parser.add_argument('--utility_test_size', type=int, required=True)

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

data_csv = ['data/UNSW_NB_15_1_withCVSS_V2.csv', 
            'data/UNSW_NB_15_2_withCVSS_V2.csv',
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
tu = [1 for i in range(len(encoder_list))]

model = TabMT(width=args.width, 
              depth=args.depth, 
              heads=args.heads, 
              encoder_list=encoder_list,
              dropout=args.dropout, 
              tu=tu)

model = fit(model=model, 
            dataset=dataset, 
            target='cvss',
            num_clusters=args.num_clusters, 
            train_size=args.train_size, 
            lr=args.lr, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            weight_decay=args.weight_decay,
            save_to_wandb=args.save_to_wandb, 
            savename=args.savename)

model.eval()

num_exp = 5
avg_acc, avg_mf1, avg_wf1 = np.zeros(num_exp), np.zeros(num_exp), np.zeros(num_exp)

for exp in range(num_exp):
    acc, mf1, wf1 = compute_catboost_utility(model=model, 
                                             frame=dataset.get_frame(), 
                                             target_name='cvss', 
                                             names=dataset.names, 
                                             dtypes=dataset.dtypes, 
                                             encoder_list=encoder_list, 
                                             label_idx=dataset.label_idx, 
                                             train_size=args.utility_train_size, 
                                             test_size=args.utility_test_size)
    
    avg_acc[exp], avg_mf1[exp], avg_wf1[exp] = acc, mf1, wf1
    
std_acc, std_mf1, std_wf1 = np.std(avg_acc), np.std(avg_mf1), np.std(avg_wf1)
avg_acc, avg_mf1, avg_wf1 = np.mean(avg_acc), np.mean(avg_mf1), np.mean(avg_wf1)

print(f'Accuracy: mean={avg_acc} std={std_acc}.')
print(f'Macro F1: mean={avg_mf1} std={std_mf1}.')
print(f'Weighted F1: mean={avg_wf1} std={std_wf1}.')