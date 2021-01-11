#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import torchtuples as tt

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models import DeepHit
from pycox.evaluation import EvalSurv


# In[2]:


np.random.seed(1234)
_ = torch.manual_seed(1234)


# In[3]:


df_train = pd.read_csv("./data/007-data_crs_train_py.csv")
df_test = pd.read_csv("./data/007-data_crs_test_py.csv")


# In[4]:


get_x = lambda df: (df
                    .drop(columns=['time', 'crstatus'])
                    .values.astype('float32'))


# In[5]:


x_train = get_x(df_train)
x_test = get_x(df_test)


# In[6]:


class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


# In[7]:


num_durations = 108
labtrans = LabTransform(num_durations)
get_target = lambda df: (df['time'].values, df['crstatus'].values)


# In[8]:


y_train = labtrans.fit_transform(*get_target(df_train))
y_test = labtrans.fit_transform(*get_target(df_test))
durations_test, events_test = get_target(df_test)


# In[9]:


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)
    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out


# In[30]:


in_features = x_train.shape[1]
num_risks = y_train[1].max()
out_features = len(labtrans.cuts)
dropout = [0.0]


# In[24]:


list_num_nodes_shared = [[32, 8], [16, 8], [16, 4], [8, 4], [24, 16, 6]]
list_num_nodes_indiv = [[32], [16], [8]]
list_batch_norm = [False]
list_lr = [0.01, 0.001, 0.001]
list_alpha = [0.1, 0.2, 0.3, 0.4]
list_sigma = [0.1, 0.2, 0.3, 0.4]
list_batch_size = [128, 256]


# In[25]:


parameters = []
for num_nodes_shared in list_num_nodes_shared:
    for num_nodes_indiv in list_num_nodes_indiv:
        for batch_norm in list_batch_norm:
            for lr in list_lr:
                for alpha in list_alpha:
                    for sigma in list_sigma:
                        for batch_size in list_batch_size:
                            parameters.append([num_nodes_shared, num_nodes_indiv, batch_norm, lr, alpha, sigma, batch_size])


# In[26]:


deephit_cv_results = pd.DataFrame(parameters)
deephit_cv_results["cindex"] = 0


# In[27]:


kf = KFold(n_splits = 5)


# In[28]:


deephit_cv_results.shape


# In[37]:


for index in range(deephit_cv_results.shape[0]):
    num_nodes_shared = deephit_cv_results.iloc[index, 0]
    num_nodes_indiv = deephit_cv_results.iloc[index, 1]
    batch_norm = deephit_cv_results.iloc[index, 2]
    lr = deephit_cv_results.iloc[index, 3]
    alpha = deephit_cv_results.iloc[index, 4]
    sigma = deephit_cv_results.iloc[index, 5]
    batch_size = deephit_cv_results.iloc[index, 6]
    
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
        
        # net = SimpleMLP(in_features, num_nodes_shared, num_risks, out_features)
        net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, 
                               num_risks, out_features, batch_norm, dropout)
        optimizer = tt.optim.AdamWR(lr = lr, decoupled_weight_decay = 0.01,
                            cycle_eta_multiplier=0.8)
        # optimizer = tt.optim.Adam(lr = lr)
        model = DeepHit(net, optimizer, alpha = alpha, sigma = sigma,
                        duration_index = labtrans.cuts)

        epochs = 100
        callbacks = [tt.callbacks.EarlyStoppingCycle()]
        verbose = False # set to True if you want printout

        log = model.fit(X_tr, Y_tr, int(batch_size), epochs,
                callbacks, verbose, val_data = (X_val, Y_val))
        
        cif = model.predict_cif(x_test)
        cif1 = pd.DataFrame(cif[0], model.duration_index)
        ev1 = EvalSurv(1-cif1, durations_test, events_test == 1, censor_surv='km')
        c_index = ev1.concordance_td()
        
        # ibs = ev1.integrated_brier_score(np.linspace(0, durations_test.max(), 100))

        cindexes.append(c_index)
        # ibss.append(ibs)

    deephit_cv_results.iloc[index, 7] = np.mean(cindexes)
    deephit_cv_results.to_csv('./data/cv.results.deephit.csv', index = False)
    # deephit_cv_results.iloc[index, 7] = c_index
    # list_ibs.append(np.mean(ibss))
    # print(parameter, np.mean(cindexes), np.mean(ibss))
    # print(index, np.mean(cindexes))
    print(deephit_cv_results.iloc[index, ].values)


# In[ ]:




deephit_cv_results = pd.read_csv("./data/cv.results.deephit.csv")
print(deephit_cv_results["cindex"].values.max())
ind_best = deephit_cv_results["cindex"].values.argmax()

num_nodes_shared = eval(deephit_cv_results.iloc[ind_best, 0])
num_nodes_indiv = eval(deephit_cv_results.iloc[ind_best, 1])
batch_norm = deephit_cv_results.iloc[ind_best, 2]
lr = deephit_cv_results.iloc[ind_best, 3]

alpha = deephit_cv_results.iloc[ind_best, 4]
sigma = deephit_cv_results.iloc[ind_best, 5]
batch_size = deephit_cv_results.iloc[ind_best, 6]




net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks, out_features, batch_norm, dropout)
optimizer = tt.optim.AdamWR(lr = lr, decoupled_weight_decay = 0.01, cycle_eta_multiplier=0.8)
model = DeepHit(net, optimizer, alpha = alpha, sigma = sigma, duration_index = labtrans.cuts)
        



    


epochs = 100
callbacks = [tt.callbacks.EarlyStoppingCycle()]
log = model.fit(x_train, y_train, int(batch_size), epochs, callbacks, val_data = (x_test, y_test))

cif = model.predict_cif(x_test)
cif1 = pd.DataFrame(cif[0], model.duration_index)
ev = EvalSurv(1-cif1, durations_test, events_test == 1, censor_surv='km')

# ev = EvalSurv(surv, Y_val_0, Y_val_1, censor_surv='km')
c_index = ev.concordance_td('antolini')
print('C-index: {:.4f}'.format(c_index))
# 0.8251

time_grid = np.linspace(df_test["time"].values.min(), df_test["time"].values.max(), 100)
ibs = ev.integrated_brier_score(time_grid) 
print('IBS: {:.4f}'.format(ibs))
# 0.1445



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
    net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks, out_features, batch_norm, dropout)
    optimizer = tt.optim.AdamWR(lr = lr, decoupled_weight_decay = 0.01, cycle_eta_multiplier=0.8)
    model = DeepHit(net, optimizer, alpha = alpha, sigma = sigma, duration_index = labtrans.cuts)
    epochs = 100
    callbacks = [tt.callbacks.EarlyStoppingCycle()]
    log = model.fit(x_train, y_train, int(batch_size), epochs, callbacks, val_data = (x_test, y_test))
    cif = model.predict_cif(x_test)
    cif1 = pd.DataFrame(cif[0], model.duration_index)
    ev = EvalSurv(1-cif1, durations_test, events_test == 1, censor_surv='km')
    # ev = EvalSurv(surv, Y_val_0, Y_val_1, censor_surv='km')
    c_index = ev.concordance_td('antolini')
    time_grid = np.linspace(df_test["time"].values.min(), df_test["time"].values.max(), 108)
    ibs = ev.integrated_brier_score(time_grid) 
    c_indexes.append(np.round(c_index, 4))
    ibss.append(np.round(ibs, 4))


pd.DataFrame(data = {"cindex": c_indexes, "ibs": ibss}).to_csv("./data/results.ci.deephit.csv", index=False)

# Compute the 95% confidence interval: conf_int
mean_cindex = np.mean(c_indexes)
mean_ibs = np.mean(ibss)


# Print the mean
print('mean cindex =', mean_cindex)
# 0.8245
print('mean ibs =', mean_ibs)
# 0.1431


ci_cindex = np.percentile(c_indexes, [2.5, 97.5])
ci_ibs = np.percentile(ibss, [2.5, 97.5])

# Print the confidence interval
print('confidence interval =', ci_cindex)
# [0.8229, 0.8255]
print('confidence interval =', ci_ibs)
# [0.1413, 0.1457]







# In[ ]:




