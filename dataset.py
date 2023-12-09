import torch
from torch.utils.data import Dataset

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

from tqdm import tqdm
import os

def unique_non_null(frame):
    return frame.dropna().unique()

class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, num_clusters): 
        self.num_clusters = num_clusters
        category_map = {'nominal': 'str',
                       'integer': 'int',
                       'binary': 'int',
                       'timestamp': 'int'}
        
        self.feat_legend = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl')
        dtypes = self.feat_legend['Type '].str.lower()
        dtypes = dtypes.replace(category_map)
        dtypes = {i: x for i, x in enumerate(dtypes)}

        self.raw_frame = self.read_files(data_csv)
        for col in self.raw_frame.columns:
            col = col.astype(dtypes[i], errors='ignore')
            idx = col.apply(lambda x: isinstance(x, eval(dtypes[i])))
            self.raw_frame = self.raw_frame[idx]

        if os.path.isfile('cache/data.csv'):
            self.frame, self.cat_dicts, self.kmeans = self.recover_from_cache()
        else:
            self.frame, self.cat_dicts, self.kmeans = self.process_data(dtypes)
            
            pickle.dump(self.cat_dicts, open("cache/cat_dicts.pkl", "wb"))
            pickle.dump(self.kmeans, open("cache/kmeans.pkl", "wb"))
            self.frame.to_csv('cache/data.csv')
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        item = self.frame.iloc[idx]
        return torch.tensor(item, dtype=torch.int)

    def read_files(self, data_csv):
        frame = pd.DataFrame()
        for i in range(len(data_csv)):
            df = pd.read_csv(data_csv[i], header=0)
            frame = pd.concat([frame, df], axis=0)

        frame = frame.sample(frac=0.3, random_state=42)
        return frame

    def process_data(dtypes):
        frame, cat_dicts, kmeans_dict = self.raw_frame.copy(), {}, {}
        for i, col in enumerate(tqdm(self.frame.columns, desc="Processing Data")):
            type = self.feat_legend.loc[idx, 'Type '].str.tolower()
            continuous = (type != 'nominal') and (type != 'binary')
            
            if (~continuous):
                unique = unique_non_null(col)
                table = dict(zip(unique, range(len(unique))))
                cat_dicts[i] = table
                col.map(table, na_action='ignore')

            elif (continuous):
                unique = len(unique_non_null(col))
                if (len(unique) > self.num_clusters):
                    kmeans, labels = self.quantizer(col, self.num_clusters)
                else:
                    kmeans, labels = self.quantizer(col, len(unique))
                
                kmeans_dict[i] = kmeans
                col = labels
            
            frame.fillna(value=-1, inplace=True)
            return frame, cat_dicts, kmeans_dicts

    def recover_from_cache():
        frame = pd.read_csv('cache/data.csv', header=0)
        cat_dicts = pickle.load(open("cache/cat_dicts.pkl", "rb"))
        kmeans = pickle.load(open("cache/kmeans.pkl", "rb"))
        return frame, cat_dicts, kmeans
    
    def quantizer(self, x, num_clusters):
        x = np.array(x, dtype=float).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(x)
        return kmeans, kmeans.labels_

    def get_categorical_dicts(self):
        dict = self.cat_dicts
        bin_idx = np.where(self.feat_legend['Type '].str.lower() == 'binary')[0]
        for idx in bin_idx:
            dict[idx] = {0: 0, 1: 1}
        return dict

    def get_occs(self):
        return self.occs