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

def replace_values_with_weights(frame):
    for ft, name in enumerate(frame.columns):
        frame.iloc[:, ft] = frame.iloc[:, ft].map(log_prob(frame.iloc[:, ft]))
    return frame.to_numpy(dtype=float)

def random_weighted_sampler(frame, size, replacement, exclude):
    num_ft, samples, masks = frame.shape[1], [], []
    frame[exclude, :] = 0.

    for batch in tqdm(range(size), desc='Sampling'):
        
        mask = torch.rand(num_ft, dtype=float).round()
        select = torch.multinomial(mask, num_samples=1)
        
        weights = torch.from_numpy(frame[:, select])
        idx = torch.multinomial(weights, num_samples=1)
        
        if not replacement:
            frame[idx, :] = 0.
        
        idx = idx.squeeze().tolist()
        mask = mask.int()
        
        if (not replacement and idx in samples) or (idx in exclude):
            continue
            
        samples.append(idx)
        masks.append(mask)
    
    return samples, torch.stack(masks, dim=0)

class UNSW_NB15_Distilled(Dataset):
    def __init__(self, dataset, size, replacement, excluded_rows, name, fixed_mask, force_refresh):
        self.size = size
        self.replacement = replacement
        self.path = f'processed_data/{name}/'
        self.refreshed = False
        self.fixed_mask = fixed_mask
        
        cache = self.cache_check() and not force_refresh
        if (cache):
            self.idx, self.masks = self.cache_read()
            print('Recovered distilled set from cache!')
        else:
            weights = replace_values_with_weights(dataset.get_frame())
            self.idx, self.masks = random_weighted_sampler(frame=weights, 
                                                           size=self.size, 
                                                           replacement=self.replacement, 
                                                           exclude=excluded_rows)
            
            self.cache_dump()
            self.refreshed = True
        
        self.frame = dataset.get_frame().iloc[self.idx]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, idx):
        num_ft = len(self.frame.columns)
        item = self.frame.iloc[idx]
        mask = self.masks[idx] if self.fixed_mask else torch.rand(num_ft).round().int()
        return torch.tensor(item, dtype=torch.int), mask
    
    def get_indices(self):
        return self.idx
    
    def get_masks(self):
        return self.masks
    
    def was_refreshed(self):
        return self.refreshed
    
    def cache_check(self):
        meta = {"size": None, "replacement": None}
        if os.path.exists(self.path + 'meta.pkl'):
            meta = pickle.load(open(self.path + 'meta.pkl', "rb"))
        exists = meta['replacement'] == self.replacement and meta['size'] == self.size
        return exists
    
    def cache_read(self):
        masks = pickle.load(open(self.path + 'masks.pkl', "rb"))
        idx = pickle.load(open(self.path + 'idx.pkl', "rb"))
        return idx, masks
    
    def cache_dump(self):
        os.makedirs(self.path, exist_ok=True)
        
        meta = {"replacement": self.replacement, "size": self.size}
        pickle.dump(meta, open(self.path + 'meta.pkl', "wb"))
        pickle.dump(self.masks, open(self.path + 'masks.pkl', "wb"))
        pickle.dump(self.idx, open(self.path + 'idx.pkl', "wb"))
        return None

def validate_split(dataset, idx, masks=None):
    frame, num_bad, avg_missing = dataset.get_frame(), 0, 0
    for ft, name in enumerate(frame.columns):
        column = frame.iloc[idx, ft]
        
        if masks != None:
            to_fill = masks[:, ft] == 1
            column = column.iloc[to_fill] 
        
        unique_in_idx = len(column.unique())
        total_unique = len(frame.iloc[:, ft].unique())
        
        if (unique_in_idx < total_unique):
            # print(f'WARNING! Only {unique_in_idx} / {total_unique} values of feature {ft} in the set.')
            num_bad += 1
            avg_missing += total_unique - unique_in_idx
    
    avg_missing = avg_missing / num_bad if (num_bad > 0) else avg_missing
    return num_bad, avg_missing
    
class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, num_clusters, drop, never_mask): 
        self.num_clusters, self.drop, self.refreshed = num_clusters, drop, False
        
        cache = self.cache_check(data_csv, num_clusters, drop)
        if (cache):
            self.frame, self.cat_dicts, self.clstr_cntrs = self.cache_read()
            print('Recovered data from cache!')
        else:
            raw_frame, dtypes, names = self.read_files(data_csv, dtype_xlsx, drop)
            cured_frame = self.cure_frame(raw_frame, dtypes)
            self.frame, self.cat_dicts, self.clstr_cntrs = self.process_data(cured_frame, dtypes, names)
            
            self.cache_dump(data_csv, num_clusters, drop)
            self.refreshed = True
            
        self.never_mask = [list(self.frame.columns).index(x) for x in never_mask]

    def read_files(self, data_csv, dtype_xlsx, drop):
        feat_legend = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl')
        names = feat_legend['Name'].str.lower().tolist()
        dtypes = feat_legend['Type '].str.lower().tolist()
        
        frame = pd.DataFrame()
        for i in range(len(data_csv)):
            df = pd.read_csv(data_csv[i], header=0, names=names, dtype=object)
            frame = pd.concat([frame, df], axis=0)
            
        frame.loc[frame['cvss'] == 'Normal', 'attack_cat'] = 'FalsePositive'
        
        for name in drop:
            dtypes.pop(names.index(name))
            names.pop(names.index(name))
        frame = frame.drop(drop, axis=1)        
        return frame, dtypes, names

    def cure_frame(self, frame, dtypes):
        numericals = np.where(np.array(dtypes) != 'nominal')[0]
        for idx in tqdm(numericals, desc="Curing Data"):
            frame.iloc[:, idx] = pd.to_numeric(frame.iloc[:, idx], errors='coerce')
        return frame

    def process_data(self, frame, dtypes, names):
        cat_dicts, clstr_cntrs = {}, {}
        for i, ft in enumerate(tqdm(frame.columns, desc="Processing Data")):
            dtype, name = dtypes[i], names[i]
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
    
    def cache_check(self, data_csv, num_clusters, drop):
        meta = {"num_clusters": None, "data_csv": None, "drop": None}
        if os.path.exists('processed_data/meta.pkl'):
            meta = pickle.load(open("processed_data/meta.pkl", "rb"))
        
        exists = meta['num_clusters'] == num_clusters and meta['data_csv'] == data_csv and meta['drop'] == drop
        return exists
    
    def cache_dump(self, data_csv, num_clusters, drop):
        meta = {"num_clusters": num_clusters, "data_csv": data_csv, "drop": drop}
        pickle.dump(meta, open("processed_data/meta.pkl", "wb"))
        pickle.dump(self.cat_dicts, open("processed_data/cat_dicts.pkl", "wb"))
        pickle.dump(self.clstr_cntrs, open("processed_data/clstr_cntrs.pkl", "wb"))
        self.frame.to_csv('processed_data/data.csv', index=False)
        return None

    def cache_read(self):
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
        num_ft = len(self.frame.columns)
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        
        item = self.frame.iloc[idx]        
        mask = torch.rand(num_ft).round().int()
        mask[self.never_mask] = 0
        return torch.tensor(item, dtype=torch.int), mask
    
    def get_frame(self):
        return self.frame
    
    def was_refreshed(self):
        return self.refreshed

    def get_categorical_dicts(self):
        return self.cat_dicts

    def get_cluster_centers(self):
        return self.clstr_cntrs
    
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