import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from catboost.metrics import F1
from modules.dataset import decode_output, stratified_sample

from imblearn.metrics import geometric_mean_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def catboost_trial(train_X, train_y, test_X, test_y, cat_features):
    classifier = CatBoostClassifier(loss_function='MultiClass',
                                eval_metric='TotalF1',
                                iterations=100,
                                use_best_model=True)
    classifier.fit(
        train_X, train_y, 
        eval_set=(test_X, test_y),
        cat_features=cat_features,
        verbose=False
    )

    predictions = classifier.predict(test_X)
    macrof1 = f1_score(test_y, predictions, average='macro')
    weightedf1 = f1_score(test_y, predictions, average='weighted')
    accuracy = accuracy_score(test_y, predictions)
    macro_gmean = geometric_mean_score(test_y, predictions, average='macro')
    weighted_gmean = geometric_mean_score(test_y, predictions, average='weighted')
    return np.array([accuracy, macrof1, weightedf1, macro_gmean, weighted_gmean])

def compute_catboost_utility(model, frame, target_name, names, dtypes, encoder_list, label_idx, train_size, test_size, num_exp, num_trials):
    names, dtypes, num_ft = np.array(names), np.array(dtypes), len(names)
    
    not_missing = (frame != -1).all(axis=1)
    clean_frame = frame[not_missing]
    
    real_train_idx, real_test_idx = stratified_sample(y=clean_frame[target_name], lengths=[train_size, test_size])
    
    condition_vectors = np.array(clean_frame.iloc[real_train_idx, label_idx], dtype=int)
    condition_vectors = torch.from_numpy(condition_vectors)
    
    gen_in = torch.ones((len(real_train_idx), num_ft), dtype=int) * -1
    gen_in[:, label_idx] = condition_vectors
    gen_in = gen_in.to(device)
    
    clean_frame = decode_output(clean_frame, encoder_list)
    real_train_frame, real_test_frame = clean_frame.iloc[real_train_idx], clean_frame.iloc[real_test_idx]
    real_train_y, real_test_y = real_train_frame[target_name], real_test_frame[target_name]
    real_train_X, real_test_X = real_train_frame.drop(names[label_idx], axis=1), real_test_frame.drop(names[label_idx], axis=1)
    
    labels = names[label_idx]
    cat_features = np.where((dtypes=='binary') | (dtypes=='nominal'))[0]
    cat_features = list(set(cat_features) - set(label_idx))
    
    real_results = catboost_trial(real_train_X, real_train_y, real_test_X, real_test_y, cat_features)
    print(f'Performance of the Classifier Trained on Real Data: {real_results}.')
    
    avg_results = []
    for exp in range(num_exp):
        gen_in_hat = torch.clone(gen_in)
        synthetics = model.gen_data(gen_in_hat, batch_size=512)
        synthetics = decode_output(synthetics, encoder_list)
        
        syn_y = synthetics[target_name]
        syn_X = synthetics.drop(names[label_idx], axis=1)
    
        trial_results = []
        for trial in range(num_trials):        
            fake_results = catboost_trial(syn_X, syn_y, real_test_X, real_test_y, cat_features)
            trial_results.append(real_results - fake_results)
            
        trial_results = np.stack(trial_results)
        print(trial_results)
        
        trial_results = np.mean(trial_results, axis=0)
        print(trial_results)
        
        avg_results.append(trial_results)
    
    print(avg_results)
    
    avg_results = np.stack(avg_results)
    means = np.mean(avg_results, axis=0)
    stds = np.std(avg_results, axis=0)
    return means, stds