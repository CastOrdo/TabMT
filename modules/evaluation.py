import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score

from catboost import CatBoostClassifier
from catboost.metrics import F1

from modules.dataset import decode_output, stratified_sample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def classifier_metrics(predictions, truths):
    macrof1 = f1_score(truths, predictions, average='macro')
    weightedf1 = f1_score(truths, predictions, average='weighted')
    accuracy = accuracy_score(truths, predictions)
    macro_gmean = geometric_mean_score(truths, predictions, average='macro')
    weighted_gmean = geometric_mean_score(truths, predictions, average='weighted')
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
    results = classifier_metrics(predictions, test_y)
    return results

def catboost_utility(synthetics, real_train, real_test, target, labels, cat_features, trials_per_syn):
    real_train_y, real_test_y = real_train[target], real_test[target]
    real_train_X, real_test_X = real_train.drop(labels, axis=1), real_test.drop(labels, axis=1)
    
    seeds = torch.randint(high=100000, size=(trials_per_syn,))
    
    results = []
    for synthetic in synthetics:
        syn_y = synthetic[target]
        syn_X = synthetic.drop(labels, axis=1)
        
        batch_results = []
        for trial in range(trials_per_syn):
            seed = int(seeds[trial])
            
            real_results = catboost_trial(real_train_X, real_train_y, 
                                          real_test_X, real_test_y, 
                                          cat_features, seed)
            fake_results = catboost_trial(syn_X, syn_y, 
                                          real_test_X, real_test_y, 
                                          cat_features, seed)
            batch_results.append(real_results - fake_results)
        
        batch_results = np.stack(batch_results)
        batch_results = np.mean(batch_results, axis=0)
        results.append(batch_results)
        
    results = np.stack(results)
    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    return means, std

def compute_catboost_utility(model, frame, train_size, test_size, target, labels, dtypes, device, num_exp, num_trials):
    model.to(device)
    
    train_idx, test_idx = stratified_sample(y=frame[target], lengths=[train_size, test_size])
    train_frame, test_frame = frame.iloc[train_idx], frame.iloc[test_idx]

    label_idx = [i for i, n in enumerate(frame.columns) if n in labels]
    labels = train_frame.iloc[:, label_idx]
    synthetic_frames = model.generate_data(labels, label_idx, num_exp, device)

    cat_features = [i for i, n in enumerate(dtypes) if (n == 'nominal') | (n == 'binary')]
    cat_features = list(set(cat_features) - set(label_idx))
    means, stds = catboost_utility(synthetic_frames, train_frame, test_frame, target, labels, cat_features, num_trials)
    return means, stds

class Ensemble(object):
    def __init__(self):
        LogReg = LogisticRegression(random_state=42,max_iter=500) 
        SVM = svm.SVC(random_state=42,probability=True)
        DT = tree.DecisionTreeClassifier(random_state=42)
        RF = RandomForestClassifier(random_state=42)
        MLP = MLPClassifier(random_state=42,max_iter=100)
        
        self.models = [LogReg, SVM, DT, RF, MLP]
    
    def fit_eval(self, X_train, y_train, X_test, y_test):
        results = []
        for model in self.models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions = np.array(predictions, dtype=int)
            
            metrics = classifier_metrics(predictions, y_test)
            results.append(metrics)
        
        results = np.concatenate(results)
        return results

def cat_to_int(frame, cat_features, encoder_list):
    frame = frame.copy()
    for ft in cat_features:
        frame.iloc[:, ft] = encoder_list[ft].transform(frame.iloc[:, ft])
    return frame
    
def minmax_preprocessing(frame):
    scaler = MinMaxScaler()
    scaler.fit(frame)
    return scaler.transform(frame)
    
def ensemble_utility(synthetics, real_train, real_test, full_real_frame, target, labels, cat_features, encoder_list):
    all_frame = pd.concat([real_train, real_test], axis=0)
    all_frame = cat_to_int(all_frame, cat_features, encoder_list)
    
    all_y = np.array(all_frame[target], dtype=int)
    real_train_y = all_y[:len(real_train)]
    real_test_y = all_y[len(real_train):]
    
    all_X = all_frame.drop(labels, axis=1)
    all_X = minmax_preprocessing(all_X)
    real_train_X = all_X[:len(real_train)]
    real_test_X = all_X[len(real_train):]
    
    real_ensemble = Ensemble()
    real_results = real_ensemble.fit_eval(real_train_X, real_train_y, 
                                          real_test_X, real_test_y)
    
    results = []
    for synthetic in synthetics:
        synthetic = cat_to_int(synthetic, cat_features, encoder_list)
        syn_y = np.array(synthetic[target], dtype=int)
        syn_X = synthetic.drop(labels, axis=1)
        syn_X = minmax_preprocessing(syn_X)
    
        fake_ensemble = Ensemble()
        fake_results = fake_ensemble.fit_eval(syn_X, syn_y, 
                                              real_test_X, real_test_y)
        results.append(real_results - fake_results)
    
    results = np.stack(results)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    return means, stds