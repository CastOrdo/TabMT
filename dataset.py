import torch
from torch.utils.data import Dataset

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

import os
import pickle
from tqdm import tqdm

def unique_non_null(frame):
    return frame.dropna().unique()

class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, num_clusters): 
        self.num_clusters = num_clusters
        self.feat_legend = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl')
        
        # raw_frame = self.read_files(data_csv)
        # self.raw_frame = self.cure_frame(raw_frame)
        self.raw_frame = pd.DataFrame()

        if os.path.exists('processed_data/meta.pkl'):
            meta = pickle.load(open("processed_data/meta.pkl", "rb"))
        else:
            meta = {"num_clusters": None, "data_csv": None}
            
        if (meta['num_clusters'] == self.num_clusters and meta['data_csv'] == data_csv):
            self.frame, self.cat_dicts, self.clstr_cntrs = self.recover_data()
        else:
            self.frame, self.cat_dicts, self.clstr_cntrs = self.process_data()
            
            meta = {"num_clusters": self.num_clusters, "data_csv": data_csv}
            pickle.dump(meta, open("processed_data/meta.pkl", "wb"))
            pickle.dump(self.cat_dicts, open("processed_data/cat_dicts.pkl", "wb"))
            pickle.dump(self.clstr_cntrs, open("processed_data/clstr_cntrs.pkl", "wb"))
            self.frame.to_csv('processed_data/data.csv', index=False)
            

    def read_files(self, data_csv):
        frame = pd.read_csv(data_csv[0], header=0, dtype=object)
        for i in range(1, len(data_csv)):
            df = pd.read_csv(data_csv[i], header=0, names=frame.columns, dtype=object)
            frame = pd.concat([frame, df], axis=0)

        condition = frame['CVSS'] == 'Normal'
        frame.loc[condition, 'Attack category_x'] = 'FalsePositive'
        return frame

    def cure_frame(self, frame):
        dtypes = self.feat_legend['Type '].str.lower() # curing data
        numericals = np.where(dtypes != 'nominal')[0]
        for idx in tqdm(numericals, desc="Curing Data"):
            frame.iloc[:, idx] = pd.to_numeric(frame.iloc[:, idx], errors='coerce') 
        return frame

    def process_data(self):
        frame, cat_dicts, clstr_cntrs = self.raw_frame.copy(), {}, {}
        for i, ft in enumerate(tqdm(frame.columns, desc="Processing Data")):
            type = self.feat_legend.loc[i, 'Type '].lower()
            name = self.feat_legend.loc[i, 'Name']
            continuous = (type != 'nominal') and (type != 'binary') and (name != 'sport') and (name != 'dsport') 

            unique = unique_non_null(frame[ft])
            if (continuous and len(unique) > self.num_clusters):
                vals = frame[ft].dropna().astype('float')
                centers, labels = self.quantizer(vals, self.num_clusters)
                frame.loc[~frame[ft].isna(), ft] = labels
            else:
                table = dict(zip(unique, range(len(unique))))
                frame[ft] = frame[ft].map(table, na_action='ignore')

            clstr_cntrs[i] = centers if (continuous and len(unique) > self.num_clusters) else (unique if continuous else None)
            cat_dicts[i] = table if not continuous else None
        
        frame.fillna(value=-1, inplace=True)
        frame = frame.sample(frac=1, random_state=42) # shuffle
        return frame, cat_dicts, clstr_cntrs

    def recover_data(self):
        frame = pd.read_csv('processed_data/data.csv', header=0)
        cat_dicts = pickle.load(open("processed_data/cat_dicts.pkl", "rb"))
        clstr_cntrs = pickle.load(open("processed_data/clstr_cntrs.pkl", "rb"))
        return frame, cat_dicts, clstr_cntrs
    
    def quantizer(self, x, num_clusters):
        x = np.array(x, dtype=float).reshape(-1,1)
        kmeans = KMeans(num_clusters, random_state=0, n_init="auto").fit(x)
        return kmeans.cluster_centers_, kmeans.labels_

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        item = self.frame.iloc[idx].values
        return torch.tensor(item, dtype=torch.int)

    def get_categorical_dicts(self):
        return self.cat_dicts

    def get_cluster_centers(self):
        return self.clstr_cntrs

    def get_raw_frame(self):
        return self.raw_frame

    def get_processed_frame(self):
        return self.frame

    def get_legend_frame(self):
        return self.feat_legend