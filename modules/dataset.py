import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
from tqdm import tqdm

def unique_non_null(frame):
    return frame.dropna().unique()

def validate_split(dataset, idx):
    frame, num_bad, avg_missing = dataset.get_frame(), 0, 0
    for ft, name in enumerate(frame.columns):
        column = frame.iloc[idx, ft]
        
        unique_in_idx = len(column.unique())
        total_unique = len(frame.iloc[:, ft].unique())
        
        if (unique_in_idx < total_unique):
            num_bad += 1
            avg_missing += total_unique - unique_in_idx
    
    avg_missing = avg_missing / num_bad if (num_bad > 0) else 0
    return num_bad, avg_missing

class ContinuousLabelEncoder(object):
    def __init__(self, kmeans, name):
        self.classes_ = kmeans.cluster_centers_
        self.classes_ = np.squeeze(self.classes_)
        
        self.type_ = 'continuous'
        self.name_ = name
    
    def inverse_transform(self, x):
        idx = range(len(self.classes_))
        mapper = dict(zip(idx, self.classes_))
        y = x.map(mapper)
        return y
    
class CategoricalLabelEncoder(object):
    def __init__(self, label_encoder, name):
        self.le = label_encoder
        self.classes_ = self.le.classes_
        self.type_ = 'categorical'
        self.name_ = name

    def inverse_transform(self, x):
        y = self.le.inverse_transform(x)
        return y.astype('str')
    
def decode_output(x, encoder_list):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    y = pd.DataFrame(x)
    for ft, encoder in enumerate(encoder_list):
        y.iloc[:, ft] = encoder.inverse_transform(y.iloc[:, ft])
        y.rename(columns={ft: encoder.name_}, inplace=True)
    return y

def read_files(data_csv, dtype_xlsx):
    feat_legend = pd.read_excel(dtype_xlsx, header=0, engine='openpyxl')
    names = feat_legend['Name'].str.lower().tolist()
    dtypes = feat_legend['Type '].str.lower().tolist()

    frame = pd.DataFrame()
    for i in range(len(data_csv)):
        df = pd.read_csv(data_csv[i], header=0, names=names, dtype=object)
        frame = pd.concat([frame, df], axis=0)
    return frame, dtypes, names

def drop_features(frame, dtypes, names, drop_names):
    frame.drop(drop_names, axis=1, inplace=True)
    for name in drop_names:
        idx = names.index(name)
        dtypes.pop(idx)
        names.pop(idx)
    return frame, dtypes, names

def cure_frame(frame, dtypes):
    numericals = np.where(np.array(dtypes) != 'nominal')[0]
    for idx in tqdm(numericals, desc="Curing Data"):        
        frame.iloc[:, idx] = pd.to_numeric(frame.iloc[:, idx], errors='coerce')
    return frame

def process_data(frame, dtypes, names, n_clusters):
    encoder_list = []
    num_ft = len(names)
    for ft in tqdm(range(num_ft), desc='Processing Data'):
        mask, dtype, name = ~frame.iloc[:, ft].isna(), dtypes[ft], names[ft]
        
        continuous = (dtype == 'integer') or (dtype == 'float') or (dtype == 'timestamp')
        if continuous:
            n_unique = len(unique_non_null(frame.iloc[:, ft]))
            n_c = n_unique if n_unique < n_clusters else n_clusters
            
            kmeans, labels = quantizer(frame.loc[mask, name], n_c)
            frame.loc[mask, name] = labels
            
            encoder = ContinuousLabelEncoder(kmeans, name)
            
        else:
            le = LabelEncoder()
            labels = le.fit_transform(frame.loc[mask, name])
            frame.loc[mask, name] = labels
            
            encoder = CategoricalLabelEncoder(le, name)

        encoder_list.append(encoder)
    
    frame.fillna(value=-1, inplace=True)
    return frame, encoder_list
        
def quantizer(x, n_clusters):
    x = np.array(x, dtype=float).reshape(-1,1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(x)
    return kmeans, labels

def stratified_sample(y, lengths):
    indices = np.arange(len(y))
    indices, y_hat = indices[y != -1], y[y != -1]
    
    subset_indices = []
    for length in lengths:
        indices, split_idx, y_hat, _ = train_test_split(indices, y_hat, 
                                                        test_size=length, 
                                                        stratify=y_hat, 
                                                        random_state=42)
        subset_indices.append(split_idx)
    return subset_indices
    
class UNSW_NB15(Dataset):
    def __init__(self, data_csv, dtype_xlsx, n_clusters, dropped_columns, labels):
        frame, dtypes, names = read_files(data_csv, dtype_xlsx)
        
        frame.loc[frame['cvss'] == 'Normal', 'attack_cat'] = 'FalsePositive'
        
        frame, dtypes, names = drop_features(frame, dtypes, names, dropped_columns)
        frame = cure_frame(frame, dtypes)
        frame, encoder_list = process_data(frame, dtypes, names, n_clusters)
        
        self.num_ft = len(names)
        self.label_idx = [i for i, n in enumerate(names) if n in labels]
        self.frame, self.dtypes, self.names, self.encoder_list = frame, dtypes, names, encoder_list
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        
        item = torch.tensor(self.frame.iloc[idx], dtype=torch.int)
        mask = torch.rand(self.num_ft).round().int()
        mask[self.label_idx] = 0
        return item, mask
    
    def get_encoder_list(self):
        return self.encoder_list
    
    def get_frame(self):
        return self.frame
