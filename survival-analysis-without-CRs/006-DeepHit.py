#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv


np.random.seed(1234)
_ = torch.manual_seed(123)


# In[3]:


df_train = pd.read_csv("./data/007-data_os_train_py.csv")
df_test = pd.read_csv("./data/007-data_os_test_py.csv")


cols_standardize = ['age', "positivelymph"]
cols_leave = [x for x in df_train.columns.to_list() if x not in ["time", "os", "age", "positivelymph"]]
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')



num_durations = 108
labtrans = DeepHitSingle.label_transform(num_durations)
get_target = lambda df: (df['time'].values, df['os'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_test = labtrans.fit_transform(*get_target(df_test))
train = (x_train, y_train)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)



in_features = x_train.shape[1]
out_features = labtrans.out_features


list_num_nodes = [[32, 8], [16, 8], [16, 4], [8, 4]]
list_batch_norm = [False, True]
list_dropout = [0.0]
list_alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
list_sigma = [0.1, 0.2, 0.3, 0.4, 0.5]
list_batch_size = [64, 128, 256, 512, 1024]
list_lr = [0.1, 0.01, 0.001, 0.0001]

parameters = []
for num_nodes in list_num_nodes:
    for batch_norm in list_batch_norm:
        for dropout in list_dropout:
            for alpha in list_alpha:
                for sigma in list_sigma:
                    for batch_size in list_batch_size:
                        for lr in list_lr:
                            parameters.append([num_nodes, batch_norm, dropout, alpha, sigma, batch_size, lr])


deephit_cv_results = pd.DataFrame(parameters)
deephit_cv_results["cindex"] = 0



kf = KFold(n_splits = 5)


for index in range(deephit_cv_results.shape[0]):
    print(index)
    num_nodes = eval(deephit_cv_results.iloc[index, 0])
    batch_norm = deephit_cv_results.iloc[index, 1]
    dropout = deephit_cv_results.iloc[index, 2]
    alpha = deephit_cv_results.iloc[index, 3]
    sigma = deephit_cv_results.iloc[index, 4]
    batch_size = deephit_cv_results.iloc[index, 5]
    lr = deephit_cv_results.iloc[index, 6]
    cindexes = []
    for train_index, test_index in kf.split(df_train):
        # print("Train:", train_index, "Validation:",test_index)
        X_tr = x_train[train_index, ]
        X_val = x_train[test_index, ]
        Y_tr_0 = y_train[0][train_index, ]
        Y_tr_1 = y_train[1][train_index, ]
        Y_val_0 = y_train[0][test_index, ]
        Y_val_1 = y_train[1][test_index, ]
        Y_tr = (Y_tr_0, Y_tr_1)
        Y_val = (Y_val_0, Y_val_1)
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        model = DeepHitSingle(net,
                      tt.optim.Adam, 
                      alpha = alpha, 
                      sigma = sigma, 
                      duration_index = labtrans.cuts)
        model.optimizer.set_lr(lr)
        epochs = 1000
        callbacks = [tt.callbacks.EarlyStopping(patience = 3)]
        log = model.fit(X_tr, Y_tr, int(batch_size), epochs, callbacks, val_data = (X_val, Y_val))
        surv = model.predict_surv_df(X_val)
        ev = EvalSurv(surv, Y_val_0, Y_val_1, censor_surv='km')
        c_index = ev.concordance_td('antolini')
        cindexes.append(c_index)
    deephit_cv_results.iloc[index, 7] = np.mean(cindexes)
    deephit_cv_results.to_csv('./data/cv.results.deephit.csv', index = False)
    print(index, np.mean(cindexes))


deephit_cv_results = pd.read_csv("./data/cv.results.deephit.csv")
print(deephit_cv_results["cindex"].values.max())
ind_best = deephit_cv_results["cindex"].values.argmax()
num_nodes = eval(deephit_cv_results.iloc[ind_best, 0])
batch_norm = deephit_cv_results.iloc[ind_best, 1]
dropout = deephit_cv_results.iloc[ind_best, 2]
alpha = deephit_cv_results.iloc[ind_best, 3]
sigma = deephit_cv_results.iloc[ind_best, 4]
batch_size = deephit_cv_results.iloc[ind_best, 5]
lr = deephit_cv_results.iloc[ind_best, 6]

        
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
model = DeepHitSingle(net,
              tt.optim.Adam, 
              alpha = alpha, 
              sigma = sigma, 
              duration_index = labtrans.cuts)
model.optimizer.set_lr(lr)
epochs = 100
callbacks = [tt.callbacks.EarlyStopping(patience = 3)]
log = model.fit(x_train, y_train, int(batch_size), epochs, callbacks, val_data = (x_test, y_test))

surv = model.predict_surv_df(x_test)
ev = EvalSurv(surv, df_test["time"].values, df_test["os"].values, censor_surv='km')
c_index = ev.concordance_td('antolini')
print('C-index: {:.4f}'.format(c_index))

time_grid = np.linspace(df_test["time"].values.min(), df_test["time"].values.max(), 108)
ibs = ev.integrated_brier_score(time_grid) 
print('IBS: {:.4f}'.format(ibs))




def bootstrap_replicate_1d(data):
    bs_sample = np.random.choice(data,len(data))
    return bs_sample


bootstrap_R = 100
c_indexes = []
ibss = []


for i in range(bootstrap_R):
    print(i)
    train_bs_idx = bootstrap_replicate_1d(np.array(range(df_train.shape[0])))
    # Creating the X, T and E input
    X_train = x_train[train_bs_idx, ]
    T_train = y_train[0][train_bs_idx]
    E_train = y_train[1][train_bs_idx]
    Y_train = (T_train, E_train)
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    model = DeepHitSingle(net,
                  tt.optim.Adam, 
                  alpha = alpha, 
                  sigma = sigma, 
                  duration_index = labtrans.cuts)
    model.optimizer.set_lr(lr)
    epochs = 100
    callbacks = [tt.callbacks.EarlyStopping(patience = 3)]
    log = model.fit(X_train, Y_train, int(batch_size), epochs, callbacks, val_data = (x_test, y_test))
    surv = model.predict_surv_df(x_test)
    ev = EvalSurv(surv, df_test["time"].values, df_test["os"].values, censor_surv='km')
    # ev = EvalSurv(surv, Y_val_0, Y_val_1, censor_surv='km')
    c_index = ev.concordance_td('antolini')
    time_grid = np.linspace(df_test["time"].values.min(), df_test["time"].values.max(), 100)
    ibs = ev.integrated_brier_score(time_grid) 
    c_indexes.append(np.round(c_index, 4))
    ibss.append(np.round(ibs, 4))


pd.DataFrame(data = {"cindex": c_indexes, "ibs": ibss}).to_csv("./data/results.ci.deephit.csv", index=False)

# Compute the 95% confidence interval: conf_int
mean_cindex = np.mean(c_indexes)
mean_ibs = np.mean(ibss)


# Print the mean
print('mean cindex =', mean_cindex)
print('mean ibs =', mean_ibs)


ci_cindex = np.percentile(c_indexes, [2.5, 97.5])
ci_ibs = np.percentile(ibss, [2.5, 97.5])

# Print the confidence interval
print('confidence interval =', ci_cindex)
print('confidence interval =', ci_ibs)

