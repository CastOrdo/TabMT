import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from catboost.metrics import F1
from modules.dataset import decode_output, stratified_sample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_catboost_utility(model, frame, target_name, names, dtypes, encoder_list, label_idx, train_size, test_size):
    names, dtypes, num_ft = np.array(names), np.array(dtypes), len(names)
    
    not_missing = (frame != -1).all(axis=1)
    clean_frame = frame[not_missing]
    
    real_train_idx, real_test_idx = stratified_sample(y=clean_frame[target_name], lengths=[train_size, test_size])
    
    condition_vectors = np.array(clean_frame.iloc[real_train_idx, label_idx], dtype=int)
    condition_vectors = torch.from_numpy(condition_vectors)
    
    gen_in = torch.ones((len(real_train_idx), num_ft), dtype=int) * -1
    gen_in[:, label_idx] = condition_vectors
    gen_in = gen_in.to(device)
    
    synthetics = model.gen_data(gen_in, batch_size=512)
    synthetics = decode_output(synthetics, encoder_list)
    
    syn_y = synthetics[target_name]
    syn_X = synthetics.drop(names[label_idx], axis=1)
    
    clean_frame = decode_output(clean_frame, encoder_list)
    real_train_frame, real_test_frame = clean_frame.iloc[real_train_idx], clean_frame.iloc[real_test_idx]
    real_train_y, real_test_y = real_train_frame[target_name], real_test_frame[target_name]
    real_train_X, real_test_X = real_train_frame.drop(names[label_idx], axis=1), real_test_frame.drop(names[label_idx], axis=1)
    
    labels = names[label_idx]
    cat_features = np.where((dtypes=='binary') | (dtypes=='nominal'))[0]
    cat_features = list(set(cat_features) - set(label_idx))
    
    classifier = CatBoostClassifier(loss_function='MultiClass',
                                    eval_metric='TotalF1',
                                    iterations=100,
                                    use_best_model=True,
                                    random_seed=42)
    classifier.fit(
        syn_X, syn_y, 
        eval_set=(real_test_X, real_test_y),
        cat_features=cat_features,
        verbose=False
    )
    
    predictions = classifier.predict(real_test_X)
    syn_macrof1 = f1_score(real_test_y, predictions, average='macro')
    syn_weightedf1 = f1_score(real_test_y, predictions, average='weighted')
    syn_accuracy = accuracy_score(real_test_y, predictions)
    
    classifier.fit(
        real_train_X, real_train_y, 
        eval_set=(real_test_X, real_test_y),
        cat_features=cat_features,
        verbose=False
    )
    
    predictions = classifier.predict(real_test_X)
    re_macrof1 = f1_score(real_test_y, predictions, average='macro')
    re_weightedf1 = f1_score(real_test_y, predictions, average='weighted')
    re_accuracy = accuracy_score(real_test_y, predictions)    
    return [re_accuracy - syn_accuracy, re_macrof1 - syn_macrof1, re_weightedf1 - syn_weightedf1]
