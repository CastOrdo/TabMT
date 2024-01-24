from modules.dataset import stratified_sample, decode_output
from modules.model import generate_data

from catboost import CatBoostClassifier
from catboost.metrics import F1

from sklearn.metrics import f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import torch
import numpy as np

def classifier_metrics(truth, predictions):
    macrof1 = f1_score(truth, predictions, average='macro')
    weightedf1 = f1_score(truth, predictions, average='weighted')
    accuracy = accuracy_score(truth, predictions)
    macro_gmean = geometric_mean_score(truth, predictions, average='macro')
    weighted_gmean = geometric_mean_score(truth, predictions, average='weighted')
    return np.array([accuracy, macrof1, weightedf1, macro_gmean, weighted_gmean])

def syn_train_test(model, frame, target_name, label_idx, encoder_list, train_size, test_size, num_syn):    
    real_train_idx, real_test_idx = stratified_sample(y=frame[target_name], lengths=[train_size, test_size])
    
    class_columns = frame.iloc[real_train_idx, label_idx]
    synthetic_frames = generate_data(model, 
                                     class_columns, 
                                     encoder_list, 
                                     label_idx, 
                                     num_files=num_syn)
    
    frame = decode_output(frame, encoder_list)
    real_train_frame, real_test_frame = frame.iloc[real_train_idx], frame.iloc[real_test_idx]
    return synthetic_frames, real_train_frame, real_test_frame

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
    results = classifier_metrics(test_y, predictions)
    return results

def catboost_utility(synthetics, real_train, real_test, target, labels, label_idx, dtypes, num_trials):
    cat_features = np.where((dtypes == 'nominal') | (dtypes == 'binary'))[0]
    cat_features = list(set(cat_features) - set(label_idx))
    seeds = torch.randint(high=1000000, size=(num_trial,))
    
    real_train_y, real_test_y = real_train[target], real_test[target]
    real_train_X, real_test_X = real_train.drop(labels, axis=1), real_test.drop(labels, axis=1)
    
    exp_results = []
    for synthetic in synthetics: 
        syn_y = synthetic[target]
        syn_X = synthetic.drop(labels, axis=1)

        results = []
        for trial in range(num_trials):
            seed = int(seeds[trial])
            real_results = catboost_trial(real_train_X, real_train_y, 
                                          real_test_X, real_test_y, 
                                          cat_features, seed)
            fake_results = catboost_trial(syn_X, syn_y, 
                                          real_test_X, real_test_y, 
                                          cat_features, seed)
            results.append(real_results - fake_results)

        results = np.stack(results)
        results = np.mean(results, axis=0)
        exp_results.append(results)
    
    exp_results = np.stack(exp_results)
    means = np.mean(exp_results, axis=0)
    stds = np.std(exp_results, axis=0)
    return means, stds

def preprocess_for_ensemble(target_frame, reference_frame, encoder_list):
    for ft, encoder in enumerate(encoder_list):
        dtype = encoder.type_
        if dtype == 'continuous':
            transform = MinMaxScaler()
            transform.fit(reference_frame.iloc[:, ft])
        else:
            transform = encoder.le
        
        target_frame.iloc[:, ft] = transform.transform(target_frame.iloc[:, ft])
    
    return target_frame

def fetch_ensemble():
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

def ensemble_utility(synthetics, real_train, real_test, full_real_frame, target, labels, encoder_list):
    real_train = preprocess_for_ensemble(real_train, full_real_frame, encoder_list)
    real_test = preprocess_for_ensemble(real_test, full_real_frame, encoder_list)
    real_train_y, real_test_y = real_train[target], real_test[target]
    real_train_X, real_test_X = real_train.drop(labels, axis=1), real_test.drop(labels, axis=1)
    
    exp_results = []
    for synthetic in synthetics:
        synthetic = preprocess_for_ensemble(synthetic, full_real_frame, encoder_list)
        syn_y = synthetic[target]
        syn_X = synthetic.drop(labels, axis=1)
    
        batch_results = []
        ensemble = zip(fetch_ensemble(), fetch_ensemble())
        for model_real, model_fake in ensemble:
            model_real.fit(real_train_X, real_train_y)
            model_fake.fit(syn_X, syn_y)

            real_pred = model_real.predict(real_test_X)
            fake_pred = model_fake.predict(real_test_X)

            real_metrics = classifier_metrics(real_test_y, real_pred)
            fake_metrics = classifier_metrics(real_test_y, fake_pred)

            batch_results.append(real_metrics - fake_metrics)

        batch_results = np.concatenate(batch_results, axis=0)
        exp_results.append(batch_results)
        
    exp_results = np.stack(exp_results)
    means = np.mean(exp_results, axis=0)
    stds = np.std(exp_results, axis=0)
    return means, stds
    
def tsne(synthetics, real, dtypes, encoder_list, savename):
    cats = np.where((dtypes == 'nominal') | (dtypes == 'binary'))[0]
    for ft in cats:
        encoder = encoder_list[ft].le
        
        all_frames = synthetics + [real]
        for frame in all_frames:
            frame.iloc[:, ft] = encoder.transform(frame.iloc[:, ft])
    
    for i, synthetic in enumerate(synthetics):
        data = pd.concat([synthetic, real], axis=0)
        fn = TSNE(n_components=2, random_state=42)
        x, y = fn.fit_transform(data)
        
        plt.figure()
        plt.scatter(x[:len(synthetic)], y[:len(synthetic)], label='synthetic')
        plt.scatter(x[:len(synthetic)], y[:len(synthetic)], label='synthetic')
        plt.title('t-SNE of Real and Synthetic Data')
        plt.xlabel('t-SNE x')
        plt.ylabel('t-SNE y')
        plt.legend()
        plt.savefig(f'figures/{savename}_{i}.png')
    
    return None
        
# def feature_distributions(synthetics, real, names, dtypes):
#     for ft, dtype in enumerate(dtypes):
#         continuous = dtype == 'float' or dtype == 'integer' or dtype == 'timestamp'
        
#         if continuous:
#             ax = real.iloc[:, ft].plot.density(label='real')
#             for i, syn in enumerate(synthetics):
#                 syn.iloc[:, ft].plot.density(label=f'synthetic_{i}')
            
#             plt.title(f'Probability Density of {names[ft]} in Real and Synthetic Data')
#             plt.legend()
#         else:
            
        