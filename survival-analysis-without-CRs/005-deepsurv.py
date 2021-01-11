#!/usr/bin/env python
# coding: utf-8


#### 1 - Importing packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.metrics import brier_score
# from pysurvival.utils.display import integrated_brier_score
from pysurvival.utils.display import display_loss_values



train = pd.read_csv("./data/007-data_os_train.csv")
test = pd.read_csv("./data/007-data_os_test.csv")
train["flag"] = 1
test["flag"] = 0


data = pd.concat([train, test])
data["os"] = (data["os"] == 4).astype(int)


features_cat = ["race", "site", "hist", "grade", "ajcc7t", "ajcc7n", "ajcc7m", "surgery", "radiation"]
features_con = ["age", "positivelymph"]


df_dummy = pd.get_dummies(data[features_cat])
data = pd.concat([data, df_dummy], axis = 1)



train = data[data["flag"] == 1]
test = data[data["flag"] == 0]


features = df_dummy.columns.to_list() + features_con

# Creating the X, T and E input
X_train, X_test = train[features].values, test[features].values
T_train, T_test = train['time'].values, test['time'].values
E_train, E_test = train['os'].values, test['os'].values


list_structure = [[{'activation': 'Sigmoid', 'num_units': 24}, 
                   {'activation': 'Sigmoid', 'num_units': 16}, 
                   {'activation': 'Sigmoid', 'num_units': 8},],
                 [{'activation': 'Sigmoid', 'num_units': 10}, 
                   {'activation': 'Sigmoid', 'num_units': 8}, 
                   {'activation': 'Sigmoid', 'num_units': 6},],
                 [{'activation': 'Sigmoid', 'num_units': 8}, 
                   {'activation': 'Sigmoid', 'num_units': 4}, 
                   {'activation': 'Sigmoid', 'num_units': 2},],
                 [{'activation': 'Sigmoid', 'num_units': 36}, 
                   {'activation': 'Sigmoid', 'num_units': 6},],
                 [{'activation': 'Sigmoid', 'num_units': 12}, 
                   {'activation': 'Sigmoid', 'num_units': 6},],
                 [{'activation': 'Sigmoid', 'num_units': 8}, 
                   {'activation': 'Sigmoid', 'num_units': 4},],
                 [{'activation': 'Sigmoid', 'num_units': 12}, 
                   {'activation': 'Sigmoid', 'num_units': 4},],]


#### 4 - Creating an instance of the NonLinear CoxPH model and fitting the data.
list_lr = [0.1, 0.01, 0.001, 0.0001]
list_num_epochs = [500, 1000, 1500]
list_optimizer = ["adadelta", "adagrad", "adam", "adamax", "rmsprop", "sgd"]



parameters = []
for structure in list_structure:
    for lr in list_lr:
        for num_epochs in list_num_epochs:
            for optimizer in list_optimizer:
                parameters.append([structure, lr, num_epochs, optimizer])


deepsurv_cv_results = pd.DataFrame(parameters)
list_cindex = []
kf = KFold(n_splits = 5)



for parameter in parameters:
    structure = parameter[0]
    lr = parameter[1]
    num_epochs = parameter[2]
    optimizer = parameter[3]
    
    cindexes = []
    for train_index, test_index in kf.split(train):
        X_tr, X_val = X_train[train_index], X_train[test_index]
        T_tr, T_val = T_train[train_index], T_train[test_index]
        E_tr, E_val = E_train[train_index], E_train[test_index]
        
        # Building the model
        nonlinear_coxph = NonLinearCoxPHModel(structure = structure)
        nonlinear_coxph.fit(X_tr, T_tr, E_tr, l2_reg = 0, batch_normalization = False,
                            verbose = True, 
                            lr = lr, num_epochs = num_epochs, optimizer = optimizer,
                            dropout = 0.)
        
        #### 5 - Cross Validation / Model Performances
        c_index = concordance_index(nonlinear_coxph, X_val, T_val, E_val)

        cindexes.append(c_index)
    list_cindex.append(np.mean(cindexes))
    print(parameter, np.mean(cindexes))


deepsurv_cv_results["cindex"] = list_cindex
deepsurv_cv_results.to_csv("./data/deepsurv_cv_results.csv", index = False)



def integrated_brier_score(model, X, T, E, t_max=None, use_mean_point=True):
    """ The Integrated Brier Score (IBS) provides an overall calculation of 
        the model performance at all available times.
    """

    # Computing the brier scores
    times, brier_scores = brier_score(model, X, T, E, t_max, use_mean_point)

    # Getting the proper value of t_max
    if t_max is None:
        t_max = max(times)
    else:
        t_max = min(t_max, max(times))

    # Computing the IBS
    ibs_value = np.trapz(brier_scores, times)/t_max 

    return ibs_value


deepsurv_cv_results = pd.read_csv("./data/deepsurv_cv_results.csv")
print(deepsurv_cv_results["cindex"].values.max())
ind_best = deepsurv_cv_results["cindex"].values.argmax()
structure = deepsurv_cv_results.iloc[ind_best, 0]
lr = deepsurv_cv_results.iloc[ind_best, 1]
num_epochs = deepsurv_cv_results.iloc[ind_best, 2]
optimizer = deepsurv_cv_results.iloc[ind_best, 3]


# Building the model
nonlinear_coxph = NonLinearCoxPHModel(structure = eval(structure))
nonlinear_coxph.fit(X_train, T_train, E_train, l2_reg = 0, batch_normalization = False,
                    verbose = True, 
                    lr = lr, num_epochs = num_epochs, optimizer = optimizer,
                    dropout = 0.)

#### 5 - Cross Validation / Model Performances
c_index = concordance_index(nonlinear_coxph, X_test, T_test, E_test)
print('C-index: {:.4f}'.format(c_index))

ibs = integrated_brier_score(nonlinear_coxph, X_test, T_test, E_test)
print('IBS: {:.4f}'.format(ibs))


def bootstrap_replicate_1d(data):
    bs_sample = np.random.choice(data,len(data))
    return bs_sample


bootstrap_R = 100
c_indexes = []
ibss = []


for i in range(bootstrap_R):
    print(i)
    train_bs_idx = bootstrap_replicate_1d(np.array(range(train.shape[0])))
    train_bs = train.iloc[train_bs_idx, ]
    # Creating the X, T and E input
    X_train = train_bs[features].values
    T_train = train_bs['time'].values
    E_train = train_bs['os'].values
    
    # Building the model
    nonlinear_coxph = NonLinearCoxPHModel(structure = eval(structure))
    nonlinear_coxph.fit(X_train, T_train, E_train, l2_reg = 0, batch_normalization = False,
                        verbose = True, 
                        lr = lr, num_epochs = num_epochs, optimizer = optimizer,
                        dropout = 0.)

    #### 5 - Cross Validation / Model Performances
    c_index = concordance_index(nonlinear_coxph, X_test, T_test, E_test)
    c_indexes.append(np.round(c_index, 4))

    ibs = integrated_brier_score(nonlinear_coxph, X_test, T_test, E_test)
    ibss.append(np.round(ibs, 4))


pd.DataFrame(data = {"cindex": c_indexes, "ibs": ibss}).to_csv("./data/results.ci.deepsurv.csv", index=False)

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

