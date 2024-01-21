from modules.evaluation import compute_catboost_utility
from modules.dataset import UNSW_NB15
from modules.model import TabMT
import torch

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
                    n_clusters=200)

encoder_list = dataset.get_encoder_list()
tu = [1 for i in range(len(encoder_list))]

model = TabMT(width=352, 
              depth=8, 
              heads=4, 
              encoder_list=encoder_list,
              dropout=0.3, 
              tu=tu)

save_path = 'saved_models/optimized'
model.load_state_dict(torch.load(save_path))

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

print(means)
print(stds)