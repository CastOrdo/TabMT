from modules.evaluation_attempt import catboost_utility, ensemble_utility

from modules.dataset import UNSW_NB15, stratified_sample
from modules.model import TabMT, generate_data
import torch
import numpy as np
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = 'saved_models/2024-01-20_23-44-52'
num_clusters = 80
width = 416
depth = 6
heads = 4

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
                    n_clusters=num_clusters)

meta = dataset.get_meta()

model = TabMT(width, depth, heads, encoder_list=meta['encoder_list'])
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(save_path))
model = model.module
model.eval()

raw_frame = meta['raw_frame']
raw_frame = raw_frame.dropna()

y, lengths = raw_frame['cvss'], [50000, 50000]
train_idx, test_idx = stratified_sample(y, lengths)
train_frame, test_frame = raw_frame.iloc[train_idx], raw_frame.iloc[test_idx]

label_idx = [i for i, n in enumerate(meta['names']) if n in labels]
class_columns = train_frame.iloc[:, label_idx]
synthetics = generate_data(model=model, 
                           class_columns=class_columns, 
                           all_encoders=meta['encoder_list'], 
                           label_idx=label_idx,
                           device=device, 
                           num_frames=1)

for i, syn in enumerate(synthetics):
    syn.to_csv('synthetic_tabmt.csv')

# categories = ['nominal', 'binary']
# cat_features = [i for i, n in enumerate(meta['dtypes']) if n in categories]
# cat_features_catboost = list(set(cat_features) - set(label_idx))

# results = catboost_utility(synthetics=synthetics, 
#                            real_train=train_frame, 
#                            real_test=test_frame, 
#                            target='cvss', 
#                            labels=labels, 
#                            cat_features=cat_features_catboost, 
#                            trials_per_syn=5)

# print(results[0])

# print(results[1])

# results = ensemble_utility(synthetics=synthetics, 
#                            real_train=train_frame, 
#                            real_test=test_frame, 
#                            full_real_frame=raw_frame, 
#                            target='cvss', 
#                            labels=labels, 
#                            encoder_list=meta['encoder_list'],
#                            cat_features=cat_features)

# print(results[0])

# print(results[1])

# print(results[0].shape)