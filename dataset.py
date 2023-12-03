import torch
from torch.utils.data import Dataset

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, num_clusters):
        self.feats = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl')
        raw_dtypes = self.feats['Type '].str.lower()
        category_map = {'nominal': 'str',
                       'integer': 'int',
                       'binary': 'int',
                       'timestamp': 'int'}
        dtypes = raw_dtypes.replace(category_map)
        self.dtypes = {i: x for i, x in enumerate(dtypes)}

        self.frame = pd.read_csv(data_csv, header=None)
        self.num_clusters = num_clusters

        self.occs = []
        self.categorical_dicts = {}
        for idx, t in enumerate(self.dtypes):
            if (self.dtypes[t] == 'str' or self.feats['Name'][idx] == 'sport' or self.feats['Name'][idx] == 'dsport'):
                unique = self.frame.iloc[:, idx].unique()
                num_unique = len(unique)
                d = dict(zip(unique, range(num_unique)))
                
                self.categorical_dicts[idx] = d
                self.frame.iloc[:, idx] = self.frame.iloc[:, idx].map(d)
            
            elif ((self.dtypes[t] == 'float' or self.dtypes[t] == 'int') and raw_dtypes[idx] != 'binary'):
                num_unique = len(self.frame.iloc[:, idx].unique())
                n_c = num_unique if num_unique < self.num_clusters else self.num_clusters
                
                occ, labels = self.quantizer(self.frame.iloc[:, idx], n_c)
                self.occs.append(occ)
                self.frame.iloc[:, idx] = labels
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.frame.iloc[idx]
        return torch.tensor(item, dtype=torch.int)
    
    def quantizer(self, x, num_clusters):
        x = np.array(x, dtype=float).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(x)
        occ = kmeans.cluster_centers_
        labels = kmeans.labels_
        return occ, labels

    def get_categorical_dicts(self):
        dict = self.categorical_dicts
        bin_idx = np.where(self.feats['Type '].str.lower() == 'binary')[0]
        for idx in bin_idx:
            dict[idx] = {0: 0, 1: 1}
        return dict

    def get_occs(self):
        return self.occs
    
data = UNSW_NB15(data_csv='UNSW-NB15_back up/UNSW-NB15_1.csv',
                   dtype_xlsx='UNSW-NB15_back up/NUSW-NB15_features.xlsx',
                   num_clusters=10)