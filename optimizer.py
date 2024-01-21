from modules.dataset import UNSW_NB15
from modules.model import TabMT
from modules.train import fit
from modules.evaluation import compute_catboost_utility
import random
import torch
import numpy as np
import math
import optuna
from time import strftime

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

data_csv = ['data/UNSW_NB_15_1_withCVSS_V2.csv', 
            'data/UNSW_NB_15_2_withCVSS_V2.csv',
            'data/UNSW_NB_15_3_withCVSS_V2.csv',
            'data/UNSW_NB_15_4_withCVSS_V2.csv']
dtype_xlsx = 'data/NUSW-NB15_features.xlsx'
dropped_columns = ['label', 'dsport', 'sport']
labels = ['cvss', 'attack_cat']

train_size = utility_train_size = utility_test_size = 50000

model = None

def objective(trial):
    global model
    
    n_clusters = trial.suggest_int('n_clusters', low=10, high=200, step=10)
    width = trial.suggest_int('width', low=16, high=512, step=16)
    depth = trial.suggest_int('depth', low=2, high=12, step=2)
    heads = trial.suggest_categorical('heads', [1, 2, 4, 8, 16])
    dropout = trial.suggest_float("dropout", low=0.0, high=0.5, step=0.05)
    lr = trial.suggest_float("lr", low=5e-5, high=5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", low=0.001, high=0.03, log=True)
    steps = trial.suggest_int("steps", low=10000, high=40000, step=5000)
    batch_size = trial.suggest_int('batch_size', low=256, high=2048, step=256)
    
    epochs = int(steps / math.ceil(train_size / batch_size))
    
    dataset = UNSW_NB15(data_csv=data_csv, 
                        dtype_xlsx=dtype_xlsx, 
                        dropped_columns=dropped_columns,
                        labels=labels,
                        n_clusters=n_clusters)

    encoder_list = dataset.get_encoder_list()
    tu = [1 for i in range(len(encoder_list))]

    model = TabMT(width=width, 
                  depth=depth, 
                  heads=heads, 
                  encoder_list=encoder_list,
                  dropout=dropout, 
                  tu=tu)
    
    savename = strftime("%Y-%m-%d_%H-%M-%S")
    model = fit(model=model, 
                dataset=dataset, 
                target='cvss',
                num_clusters=n_clusters, 
                train_size=train_size, 
                lr=lr, 
                epochs=epochs, 
                batch_size=batch_size, 
                weight_decay=weight_decay,
                savename=savename)

    model.eval()
    num_exp, trials_per_exp = 5, 5
    means, stds = compute_catboost_utility(model=model, 
                                           frame=dataset.get_frame(), 
                                           target_name='cvss', 
                                           names=dataset.names, 
                                           dtypes=dataset.dtypes, 
                                           encoder_list=encoder_list, 
                                           label_idx=dataset.label_idx, 
                                           train_size=utility_train_size, 
                                           test_size=utility_test_size,
                                           num_trials=trials_per_exp, 
                                           num_exp=num_exp)
    return means[1]

save_path = 'saved_models/optimized'
def save_model(study, trial):
    if study.best_trial == trial:
        torch.save(model.state_dict(), save_path)

study = optuna.create_study(study_name="optimizing_parameters", 
                            direction='minimize', 
                            load_if_exists=True, 
                            storage="sqlite:///optimizer_sql.db")
study.optimize(objective, n_trials=7, callbacks=[save_model])
