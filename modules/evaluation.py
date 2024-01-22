import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from catboost import CatBoostClassifier
from catboost.metrics import F1
from modules.dataset import decode_output, stratified_sample

from imblearn.metrics import geometric_mean_score

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def classifier_metrics(predictions, truth):
    macrof1 = f1_score(truth, predictions, average='macro')
    weightedf1 = f1_score(truth, predictions, average='weighted')
    accuracy = accuracy_score(truth, predictions)
    macro_gmean = geometric_mean_score(truth, predictions, average='macro')
    weighted_gmean = geometric_mean_score(truth, predictions, average='weighted')
    return np.array([accuracy, macrof1, weightedf1, macro_gmean, weighted_gmean])

def catboost_trial(train_X, train_y, test_X, test_y, cat_features, seed):    
    classifier = CatBoostClassifier(loss_function='MultiClass',
                                    eval_metric='TotalF1',
                                    iterations=100,
                                    use_best_model=True,
                                    random_seed=seed)
    classifier.fit(
        train_X, train_y, 
        eval_set=(test_X, test_y),
        cat_features=cat_features,
        verbose=False
    )

    predictions = classifier.predict(test_X)
    metrics = classifier_metrics(predictions, test_y)
    return metrics

def catboost_utility(syn_X, syn_y, real_train_X, real_train_y, real_test_X, real_test_y, num_trials, cat_features, seeds):
    trial_results = []
    for trial in range(num_trials):
        seed = int(seeds[trial])
        real_results = catboost_trial(real_train_X, real_train_y, 
                                      real_test_X, real_test_y, 
                                      cat_features, seed)
        fake_results = catboost_trial(syn_X, syn_y, 
                                      real_test_X, real_test_y, 
                                      cat_features, seed)
        trial_results.append(real_results - fake_results)
        
    trial_results = np.stack(trial_results)
    trial_results = np.mean(trial_results, axis=0)
    return trial_results

def create_weak_ensemble():
    LogReg = LogisticRegression(random_state=42,max_iter=500) 
    SVM = svm.SVC(random_state=42,probability=True)
    DT = tree.DecisionTreeClassifier(random_state=42)
    RF = RandomForestClassifier(random_state=42)
    MLP = MLPClassifier(random_state=42,max_iter=100)
    LinReg = LinearRegression()
    R = Ridge(random_state=42)
    L = Lasso(random_state=42)
    BR = BayesianRidge()
    return [LogReg, SVM, DT, RF, MLP, LinReg, R, L, BR]

def preprocess_frame(subset_frame, full_frame, encoder_list):
    subset_frame = subset_frame.copy()
    
    fts = np.where(subset_frame.columns == full_frame.columns)[0]
    for ft in fts:
        dtype = encoder_list[ft].type_
        
        if (dtype == 'continuous'):
            encoder = MinMaxScaler()
            encoder.fit(full_frame.iloc[:, ft])
        else:
            encoder = encoder_list[ft].le

        subset_frame.iloc[:, ft] = encoder.transform(subset_frame.iloc[:, ft])
    return subset_frame

def weak_ensemble_utility(syn_X, syn_y, real_train_X, real_train_y, real_test_X, real_test_y, frame, encoder_list):
    models_real, models_syn = create_weak_ensemble(), create_weak_ensemble()
    
    syn_X, syn_y = preprocess_for_ensemble(syn_X, frame, encoder_list), preprocess_for_ensemble(syn_y, frame, encoder_list)
    real_train_X, real_train_y = preprocess_for_ensemble(real_train_X, frame, encoder_list), preprocess_for_ensemble(real_train_y, frame, encoder_list)
    real_test_X, real_test_y = preprocess_for_ensemble(real_test_X, frame, encoder_list), preprocess_for_ensemble(real_test_y, frame, encoder_list)

    results = []
    for i in range(len(models_real)):
        models_real[i].fit(real_train_X, real_train_y)
        predictions = models_real[i].predict(real_test_X)
        real_results = classifier_metrics(predictions, real_test_y)

        models_syn[i].fit(syn_X, syn_y)
        predictions = models_syn[i].predict(real_test_X)
        fake_results = classifier_metrics(predictions, real_test_y)
        
        results.append(real_results - fake_results)
    
    results = np.stack(results)
    return results

def compute_utility(model, frame, target_name, names, dtypes, encoder_list, label_idx, train_size, test_size, num_exp, num_trials, weak_ensemble=False):
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
    
    seeds, avg_results = torch.randint(high=1000000, size=(num_trials,)), []
    for exp in range(num_exp):
        gen_in_hat = torch.clone(gen_in)
        synthetics = model.gen_data(gen_in_hat, batch_size=512)
        synthetics = decode_output(synthetics, encoder_list)
        
        syn_y = synthetics[target_name]
        syn_X = synthetics.drop(names[label_idx], axis=1)
    
        cat_util = catboost_utility(syn_X, syn_y, 
                                    real_train_X, real_train_y, 
                                    real_test_X, real_test_y, 
                                    num_trials, 
                                    cat_features, 
                                    seeds)
        print(cat_util)

        if weak_ensemble:
            ensemble_util = weak_ensemble_utility(syn_X, syn_y, 
                                                  real_train_X, real_train_y, 
                                                  real_test_X, real_test_y, 
                                                  frame, encoder_list)
            print(ensemble_util)
        
        results = np.concatenate((cat_util, ensemble_util), axis=0) if weak_ensemble else cat_util
        avg_results.append(results)
    
    avg_results = np.stack(avg_results)
    means = np.mean(avg_results, axis=0)
    stds = np.std(avg_results, axis=0)
    return means, stds