import torch
from torch.utils.data import Dataset

from sklearn.cluster import KMeans

import pandas as pd
import numpy as np

import math
import os
import pickle
from tqdm import tqdm

def unique_non_null(frame):
    return frame.dropna().unique()

def log_prob(col):
    return dict(np.log(col.value_counts(normalize=True)).abs())

class UNSW_NB15_Distilled(Dataset):
    def __init__(self, raw_dataset, size, replacement):
        self.frame = raw_dataset.get_processed_frame()
        self.replacement = replacement
        self.size = size
        
        meta = {"replacement": None, "size": None}
        if os.path.exists('distilled_data/meta.pkl'):
            meta = pickle.load(open("distilled_data/meta.pkl", "rb"))
        
        dataset_refreshed = raw_dataset.was_dataset_refreshed()
        if (meta['replacement'] == self.replacement and meta['size'] == size and dataset_refreshed == False):
            self.frame = pd.read_csv('distilled_data/data.csv', header=0, dtype=int)
            self.masks = pickle.load(open("distilled_data/masks.pkl", "rb"))
            print('Recovered Distilled Data from Cache!')
            
            
        else:
            self.probs = self.frame.copy()
            for ft in range(len(self.probs.columns)):
                self.probs.iloc[:, ft] = self.probs.iloc[:, ft].map(log_prob(self.probs.iloc[:, ft]))
            self.probs = self.probs.to_numpy(dtype=np.int8)
        
            self.frame, self.masks = self.selective_sample()
            
            meta = {"replacement": self.replacement, "size": self.size}
            pickle.dump(meta, open("distilled_data/meta.pkl", "wb"))
            pickle.dump(self.masks, open("distilled_data/masks.pkl", "wb"))
            self.frame.to_csv('distilled_data/data.csv', index=False) 

    def selective_sample(self):
        samples, masks = [], []
        num_ft = len(self.frame.columns)
        
        batch_size = 1000
        batches = math.ceil(self.size / batch_size)
        for batch in tqdm(range(batches), desc='Distilling Dataset'):
            
            m = torch.rand((batch_size, num_ft)).round().int()
            fts = torch.multinomial(m.float(), num_samples=1, replacement=True)
            w = torch.from_numpy(self.probs[:, fts.squeeze()])
            w = torch.transpose(w, dim0=0, dim1=1)
            
            s = torch.multinomial(w.float(), num_samples=1, replacement=self.replacement)
            s = s.squeeze().tolist()
            
            samples.extend(s)
            masks.append(m)
            if not self.replacement:
                self.probs[s, :] = 0
        
        masks = torch.cat(masks, dim=0)
        frame = self.frame.iloc[samples, :]
        return frame, masks
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        row = self.frame.iloc[idx]
        mask = self.masks[idx]
        return torch.tensor(row, dtype=torch.int), mask

class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, num_clusters, drop): 
        self.num_clusters = num_clusters
        self.drop = drop
        self.refreshed = False
        
        meta = {"num_clusters": None, "data_csv": None, "drop": None}
        if os.path.exists('processed_data/meta.pkl'):
            meta = pickle.load(open("processed_data/meta.pkl", "rb"))
            
        if (meta['num_clusters'] == self.num_clusters and meta['data_csv'] == data_csv and meta['drop'] == drop):
            self.frame, self.cat_dicts, self.clstr_cntrs = self.recover_data()
            print('Recovered data from cache!')
            
        else:
            raw_frame, dtypes, names = self.read_files(data_csv, dtype_xlsx, drop)
            cured_frame = self.cure_frame(raw_frame, dtypes)
            self.frame, self.cat_dicts, self.clstr_cntrs = self.process_data(cured_frame, dtypes, names)
            self.refreshed = True
            
            meta = {"num_clusters": self.num_clusters, "data_csv": data_csv, "drop": drop}
            pickle.dump(meta, open("processed_data/meta.pkl", "wb"))
            pickle.dump(self.cat_dicts, open("processed_data/cat_dicts.pkl", "wb"))
            pickle.dump(self.clstr_cntrs, open("processed_data/clstr_cntrs.pkl", "wb"))
            self.frame.to_csv('processed_data/data.csv', index=False)

    def read_files(self, data_csv, dtype_xlsx, drop):
        feat_legend = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl')
        names = feat_legend['Name'].str.lower().tolist()
        dtypes = feat_legend['Type '].str.lower().tolist()
        
        frame = pd.DataFrame()
        for i in range(len(data_csv)):
            df = pd.read_csv(data_csv[i], header=0, names=names, dtype=object)
            frame = pd.concat([frame, df], axis=0)
            
        frame.loc[frame['cvss'] == 'Normal', 'attack_cat'] = 'FalsePositive'
        
        frame = frame.drop(drop, axis=1)
        for name in drop:
            idx = names.index(name)
            names.pop(idx)
            dtypes.pop(idx)
        return frame, dtypes, names

    def cure_frame(self, frame, dtypes):
        numericals = np.where(np.array(dtypes) != 'nominal')[0]
        for idx in tqdm(numericals, desc="Curing Data"):
            frame.iloc[:, idx] = pd.to_numeric(frame.iloc[:, idx], errors='coerce') 
        return frame

    def process_data(self, raw_frame, dtypes, names):
        frame, cat_dicts, clstr_cntrs = raw_frame.copy(), {}, {}
        for i, ft in enumerate(tqdm(frame.columns, desc="Processing Data")):
            dtype = dtypes[i]
            name = names[i]
            continuous = (dtype != 'nominal') and (dtype != 'binary') and (name != 'sport') and (name != 'dsport') 

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
        return frame, cat_dicts, clstr_cntrs

    def recover_data(self):
        frame = pd.read_csv('processed_data/data.csv', header=0, dtype=int)
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
        item = self.frame.iloc[idx]
        return torch.tensor(item, dtype=torch.int)

    def get_categorical_dicts(self):
        return self.cat_dicts

    def get_cluster_centers(self):
        return self.clstr_cntrs

    def get_processed_frame(self):
        return self.frame
    
    def was_dataset_refreshed(self):
        return self.refreshed

class ReverseTokenizer():
    def __init__(self, cat_dicts, clstr_cntrs, num_ft):
        self.num_ft = num_ft
        self.reverse_table = {}
        for ft in range(num_ft):
            if (cat_dicts[ft] != None):
                self.reverse_table[ft] = {v: k for k, v in cat_dicts[ft].items()}
            else:
                self.reverse_table[ft] = {zip(range(len(clstr_cntrs[ft])), clstr_cntrs[ft])}

    def decode(self, x):
        out = pd.DataFrame(x.numpy(), dtype='int')
        for ft in range(self.num_ft):
            out.iloc[:, ft] = out.iloc[:, ft].replace(self.reverse_table[ft])
        return out