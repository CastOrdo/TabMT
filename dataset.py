import torch
from torch.utils.data import Dataset

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

from tqdm import tqdm

class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, num_clusters):        
        self.feats = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl').drop(47, axis=0)
        raw_dtypes = self.feats['Type '].str.lower()
        category_map = {'nominal': 'str',
                       'integer': 'int',
                       'binary': 'int',
                       'timestamp': 'int'}
        dtypes = raw_dtypes.replace(category_map)
        self.dtypes = {i: x for i, x in enumerate(dtypes)}

        self.frame = self.read_files(data_csv)
        self.frame = self.frame.sample(frac=0.3, random_state=42) # shuffle
        self.num_clusters = num_clusters

        self.occs = []
        self.categorical_dicts = {}
        for idx, t in enumerate(tqdm(self.dtypes, desc="Processing Data", unit="column")):

            # resolving missing values
            missing = self.frame.iloc[:, idx].isna()
            column = self.frame[~missing].iloc[:, idx]
            if sum(missing != 0):
                self.frame.loc[missing, self.frame.columns[idx]] = -1
                print(f'\nResolved {sum(missing)} missing values in column {idx}.')
            
            # enforcing dtypes and dropping bad rows
            column = column.astype(self.dtypes[idx], errors='ignore')
            good_indices = column.apply(lambda x: isinstance(x, eval(self.dtypes[idx])))
            if (sum(good_indices) < len(column)):
                print(f'\nScrapping {sum(~good_indices)} rows due to bad values in column {idx}.')
                self.frame = self.frame[good_indices]
                missing = missing[good_indices]
                column = column[good_indices]

            # tokenizing categorical columns
            if (self.dtypes[t] == 'str'):
                unique = column.unique()
                num_unique = len(unique)
                d = dict(zip(unique, range(num_unique)))
                
                self.categorical_dicts[idx] = d
                column = column.map(d)

                self.frame.loc[~missing, self.frame.columns[idx]] = column

            # tokenizing and quantizing continuous columns
            elif ((self.dtypes[t] == 'float' or self.dtypes[t] == 'int') and raw_dtypes.iloc[idx] != 'binary'):
                num_unique = len(column.unique())
                n_c = num_unique if num_unique < self.num_clusters else self.num_clusters
                
                occ, labels = self.quantizer(column, n_c)
                self.occs.append(occ)
                column = labels

                self.frame.loc[~missing, self.frame.columns[idx]] = column
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.frame.iloc[idx]
        return torch.tensor(item, dtype=torch.int)

    def read_files(self, data_csv):
        frame = pd.read_csv(data_csv[0], header=None)
        for i in range(1, len(data_csv)):
            df = pd.read_csv(data_csv[i], header=None)
            frame = pd.concat([frame, df], axis=0)

        frame = frame.drop(47, axis=1)
        return frame
    
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