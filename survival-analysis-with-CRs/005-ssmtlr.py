#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import h5py
import scipy
from sklearn.model_selection import KFold
from scipy import ndimage
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from lifelines.utils import concordance_index
from pycox.evaluation import EvalSurv


# In[2]:

# from tensorflow import keras
# from keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as k
from sklearn.preprocessing import normalize
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

import os
import tensorflow as tf
# from tensorflow.keras.backend.tensorflow_backend import set_session

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))

np.random.seed(1234)
tf.random.set_seed(2021)



df_train = pd.read_csv("./data/007-data_crs_train_py.csv")
df_test = pd.read_csv("./data/007-data_crs_test_py.csv")



cols_standardize = ['age', "positivelymph"]
cols_leave = [x for x in df_train.columns.to_list() if x not in ["time", "crstatus", "age", "positivelymph"]]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')


# get_x = lambda df: (df
#                     .drop(columns=['time', 'crstatus'])
#                     .values.astype('float32'))
# x_train = get_x(df_train)
# x_test = get_x(df_test)






# In[6]:


def get_y_labels(status, time):
    ret = np.zeros((status.shape[0], np.max(time) + 1))
    # ret = np.ones((status.shape[0], 8))
    for i in range(status.shape[0]):
        if status[i] == 1: # csd
            ret[i, 0:time[i] - 1 + 1] = 0
            ret[i, time[i] - 1 + 1:] = 1
        elif status[i] == 2: # other death
            ret[i, 0:time[i] - 1 + 1] = 0
            ret[i, time[i] - 1 + 1:] = 2  
        elif status[i] == 0: # censor
            ret[i, 0:time[i] + 1] = 0
            ret[i, time[i] + 1:] = 3   
    return ret


# In[7]:


y_train = get_y_labels(df_train['crstatus'], df_train['time'])
y_test = get_y_labels(df_test['crstatus'], df_test['time'])

time_interval = 6
time_max = np.max(df_train['time'])
time_length = time_max//time_interval
y_train = y_train[:, np.arange(time_interval, time_max, time_interval)]
y_test = y_test[:, np.arange(time_interval, time_max, time_interval)]

print ("train_set_x shape: " + str(x_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(x_test.shape))
print ("test_set_y shape: " + str(y_test.shape))


# In[9]:


y_train_status = to_categorical(y_train)
y_test_status = to_categorical(y_test)


# In[11]:

def reshape_y(y):
    dim = y.shape[1]
    ret = []
    for i in range(dim):
        ret.append(y[:, i, 0:3])
    return ret        


# In[12]:


y_train_status = reshape_y(y_train_status)
y_test_status = reshape_y(y_test_status)


# In[13]:


y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)


# In[14]:


y_train_status_f = y_train_status + [y_train] + [y_train]
y_test_status_f = y_test_status + [y_test] + [y_test]


# In[15]:

get_target = lambda df: (df['time'].values, df['crstatus'].values)
durations_train, events_train = get_target(df_train)
durations_test, events_test = get_target(df_test)


def logloss(lambda3):
    def loss(y_true, y_pred):
        mask_alive = y_true[:, 0] # y_true[:, np.array([0,2,4,6,8,10,12,14])]
        mask_dead_cause = y_true[:, 1] # y_true[:, np.array([1,3,5,7,9,11,13,15])]
        mask_dead_other = y_true[:, 2] # y_true[:, np.array([0,2,4,6,8,10,12,14])]
        mask_censored = 1 - (mask_alive + mask_dead_cause + mask_dead_other)
   
        logloss = -1 * k.mean(mask_alive * k.log(y_pred[:, 0]) + mask_dead_cause * k.log(y_pred[:, 1])
                         + mask_dead_other * k.log(y_pred[:, 2])) #/ 43185
        - lambda3 * k.mean(y_pred[:, 1] * mask_censored * k.log(y_pred[:, 1]) + y_pred[:, 0] * mask_censored * k.log(y_pred[:, 0])
                   + y_pred[:, 2] * mask_censored * k.log(y_pred[:, 2])) # / 39899
        return logloss
    return loss

# def logloss(lambda3):
#     def loss(y_true, y_pred):
#         mask_alive = y_true[:, 0]
#         mask_c_dead = y_true[:, 1]
#         mask_o_dead = y_true[:, 2]
#         mask_censored = 1 - (mask_alive + mask_c_dead)
#         logloss = -1 * k.mean(mask_c_dead * k.log(y_pred[:, 1]) + mask_alive * k.log(y_pred[:, 0])) 
#         - lambda3 * k.mean(y_pred[:, 1] * mask_censored * k.log(y_pred[:, 1]))
#         return logloss
#     return loss

# def logloss(lambda3):
#     def loss(y_true, y_pred):
#         mask_dead = y_true[:, 1] # y_true[:, np.array([1,3,5,7,9,11,13,15])]
#         mask_alive = y_true[:, 0] # y_true[:, np.array([0,2,4,6,8,10,12,14])]
#         mask_censored = 1 - (mask_alive + mask_dead)
#         logloss = -1 * k.mean(mask_dead * k.log(y_pred[:, 1]) + mask_alive * k.log(y_pred[:, 0])) #/ 43185
#         - lambda3 * k.mean(y_pred[:, 1] * mask_censored * k.log(y_pred[:, 1])) # / 39899
#         return logloss
#     return loss

# In[16]:


def rankingloss_cause(y_true, y_pred, name = None):
    ranking_loss = 0
    # idx = [1,3,5,7,9,11,13,15]
    for i in range(time_length):
        for j in range(i + 1, time_length, 1):
            tmp = y_pred[:, i] - y_pred[:, j]
            tmp1 = tmp > 0
            tmp1 = tf.cast(tmp1, tf.float32)
            # ranking_loss = ranking_loss + k.mean(f_loss([tmp]) * tmp)
            ranking_loss = ranking_loss + k.mean((tmp1 * tmp * (j - i) * 60))
    return ranking_loss


def rankingloss_other(y_true, y_pred, name = None):
    ranking_loss = 0
    for i in range(time_length):
        for j in range(i + 1, time_length, 1):
            tmp = y_pred[:, i] - y_pred[:, j]
            tmp1 = tmp > 0
            tmp1 = tf.cast(tmp1, tf.float32)
            # ranking_loss = ranking_loss + k.mean(f_loss([tmp]) * tmp)
            ranking_loss = ranking_loss + k.mean((tmp1 * tmp * (j - i) * 12))
    return ranking_loss


list_lambda3 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
list_lambda4 = [0.01, 0.1, 1, 10, 100]
list_lr = [0.01]
list_batch_size = [128]




parameters = []
for lambda3 in list_lambda3:
    for lambda4 in list_lambda4:
        for lr in list_lr:
            for batch_size in list_batch_size:
                parameters.append([lambda3, lambda4, lr, batch_size])


# In[26]:


ssmtlr_cv_results = pd.DataFrame(parameters)
ssmtlr_cv_results["cindex"] = 0


# In[27]:


kf = KFold(n_splits = 5)


# In[28]:


print(ssmtlr_cv_results.shape)




# for index in range(ssmtlr_cv_results.shape[0]):
#     lambda3 = ssmtlr_cv_results.iloc[index, 0]
#     lambda4 = ssmtlr_cv_results.iloc[index, 1]
#     lr = ssmtlr_cv_results.iloc[index, 2]
#     batch_size = ssmtlr_cv_results.iloc[index, 3]

#     cindexes = []
#     for train_index, test_index in kf.split(df_train):
#         X_tr = x_train[train_index, ]
#         X_val = x_train[test_index, ]

#         Y_tr_0 = y_train[train_index, ]
#         Y_val_0 = y_train[test_index, ]

#         Y_tr_1 = []
#         Y_val_1 = []
#         for i in range(time_length):
#             Y_tr_1.append(y_train_status[i][train_index])
#             Y_val_1.append(y_train_status[i][test_index])
        
#         Y_tr = Y_tr_1 + [Y_tr_0]
#         Y_val = Y_val_1 + [Y_val_0]

#         input_tensor = Input((X_tr.shape[1],))
#         x = input_tensor
#         x = Dense(24, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.))(x)
#         x = Dense(8, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.))(x)

#         prepare_list = {}
#         for i in range(time_length):
#              prepare_list['x' + str(i)] = Dense(2, activation = 'softmax', kernel_regularizer = L1L2(l1 = 0., l2 = 0.), name = 'month_' + str(i))(x)

#         xx1 = concatenate(list(prepare_list.values()))
#         xx2 = Lambda(lambda x: x[:, 1::2], name = 'ranking')(xx1)

#         losses = {}
#         loss_weights = {}
#         for i in range(time_length):
#             losses['month_' + str(i)] = logloss(lambda3)
#             loss_weights['month_' + str(i)] = 1
#         losses['ranking'] = rankingloss
#         loss_weights['ranking'] = lambda4

#         model = Model(input_tensor, list(prepare_list.values()) + [xx2])
#         model.compile(optimizer = Adam(lr),
#                       loss = losses, # 'categorical_crossentropy',
#                       loss_weights = loss_weights)
#         model.fit(X_tr, Y_tr, epochs = 100, validation_data=(X_val, Y_val), 
#                   batch_size = batch_size, shuffle = True, # verbose = 0,
#                   callbacks = [
#                   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0),
#                   # roc_callback(training_data=(X_train, Y_train_status),validation_data=(X_test, Y_test_status)),
#                      # LearningRateScheduler(lr_schedule),
#                       EarlyStopping(patience = 3)])

#         y_test_status_pred = model.predict(x_test)
#         pred = np.array(y_test_status_pred[0:time_length])
#         pred_dead = pred[:, :, 1]
#         c_index = concordance_index(durations_test, 1 - pred_dead[-1, :], event_observed = events_test)
#         cindexes.append(c_index)

#     ssmtlr_cv_results.iloc[index, 4] = np.mean(cindexes)
#     ssmtlr_cv_results.to_csv('./data/cv.results.ssmtlr.csv', index = False)
#     print(ssmtlr_cv_results.iloc[index, ].values)


lambda3 = 1
lambda4 = 0.001
lr = 0.01
batch_size = 2048

input_tensor = Input((x_train.shape[1],))
x = input_tensor
# x = Dense(16, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.),
    # kernel_initializer= tf.keras.initializers.VarianceScaling())(x)
# x = BatchNormalization()(x)
x = Dense(16, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.),
    kernel_initializer= tf.keras.initializers.VarianceScaling())(x)
x = BatchNormalization()(x)
x = Dense(8, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.),
    kernel_initializer= tf.keras.initializers.VarianceScaling())(x)
x = BatchNormalization()(x)



# In[18]:


prepare_list = {}
for i in range(time_length):
     prepare_list['x' + str(i)] = Dense(3, activation = 'softmax', kernel_regularizer = L1L2(l1 = 0., l2 = 0.), name = 'month_' + str(i))(x)

xx1 = concatenate(list(prepare_list.values()))
xx2 = Lambda(lambda x: x[:, 1::3], name = 'ranking_cause')(xx1)
xx3 = Lambda(lambda x: x[:, 2::3], name = 'ranking_other')(xx1)

losses = {}
loss_weights = {}
for i in range(time_length):
    losses['month_' + str(i)] = logloss(lambda3)
    loss_weights['month_' + str(i)] = 1

losses['ranking_cause'] = rankingloss_cause
losses['ranking_other'] = rankingloss_other
loss_weights['ranking_cause'] = lambda4
loss_weights['ranking_other'] = lambda4

# In[21]:


model = Model(input_tensor, list(prepare_list.values()) + [xx2] + [xx3])
# model = Model(input_tensor, list(prepare_list.values()) + [xx2])
model.compile(optimizer = Adam(lr),
              loss = losses, # 'categorical_crossentropy',
              loss_weights = loss_weights)

# In[ ]:


model.fit(x_train, y_train_status_f, epochs = 100, validation_data=(x_test, y_test_status_f), 
          batch_size = batch_size, shuffle = True, verbose = 1,
          callbacks = [
          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0),
          EarlyStopping(patience = 5)])


# In[ ]:


y_test_status_pred = model.predict(x_test)
pred = np.array(y_test_status_pred[0:time_length])
pred_dead = pred[:, :, 1]


# In[ ]:

# def integrated_brier_score(y_true, y_pred, times, t_max):
#     """ The Integrated Brier Score (IBS) provides an overall calculation of 
#         the model performance at all available times.
#     """
#     # Computing the brier scores
#     brier_scores = []
#     for i in range(len(times)):
#         brier_score = metrics.brier_score_loss(y_true[:, i], y_pred[:, i], pos_label = 1)
#         brier_scores.append(brier_score)
#     # Computing the IBS
#     ibs_value = np.trapz(brier_scores, times)/t_max 
#     return ibs_value
# ibs = integrated_brier_score((y_test == 1).astype(int), pred_dead.T, np.arange(1, np.max(durations_test//time_interval)), np.max(durations_test//time_interval + 1))
# print('C-index: {:.4f}'.format(c_index))
# print('IBS: {:.4f}'.format(ibs))


# In[ ]:
# ------------------------------

# cif1 = pd.DataFrame(pred_dead, np.arange(time_length) + 1)
# ev1 = EvalSurv(1-cif1, durations_test, events_test == 1, censor_surv='km')
# c_index = ev1.concordance_td()
# ibs = ev1.integrated_brier_score(np.arange(time_length))

cif1 = pd.DataFrame(pred_dead, np.arange(time_length) + 1)
ev1 = EvalSurv(1-cif1, durations_test//time_interval, events_test == 1, censor_surv='km')
c_index = ev1.concordance_td('antolini')
ibs = ev1.integrated_brier_score(np.arange(time_length)) # np.linspace(0, time_length, time_length + 1))

 # If 'method' is 'antolini', the concordance from Antolini et al. is computed.
    # If 'method' is 'adj_antolini' (default) we have made a small modifications

# results.append(max(aucs))

print('C-index: {:.4f}'.format(c_index))
# 0.8980
print('IBS: {:.4f}'.format(ibs))
# 0.0222




# ------------------------------


def bootstrap_replicate_1d(data):
    bs_sample = np.random.choice(data,len(data))
    return bs_sample


bootstrap_R = 100
c_indexes = []
ibss = []


for i in range(bootstrap_R):
    print(i)
    train_bs_idx = bootstrap_replicate_1d(np.array(range(x_train.shape[0])))
    # train_bs = train.iloc[train_bs_idx, ]
    # X_train = train_bs[features].values
    # T_train = train_bs['time'].values
    # E_train = train_bs['os'].values
    X_tr = x_train[train_bs_idx, ]
    Y_tr_0 = y_train[train_bs_idx, ]

    Y_tr_1 = []
    for i in range(time_length):
        Y_tr_1.append(y_train_status[i][train_bs_idx])
    
    Y_tr = Y_tr_1 + [Y_tr_0] + [Y_tr_0]

    input_tensor = Input((X_tr.shape[1],))
    x = input_tensor
    # x = Dense(24, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.),
    # kernel_initializer= tf.keras.initializers.VarianceScaling())(x)
    # x = BatchNormalization()(x)
    x = Dense(16, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.),
    kernel_initializer= tf.keras.initializers.VarianceScaling())(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation = 'sigmoid', kernel_regularizer = L1L2(l1 = 0., l2 = 0.),
    kernel_initializer= tf.keras.initializers.VarianceScaling())(x)
    x = BatchNormalization()(x)

    prepare_list = {}
    for i in range(time_length):
         prepare_list['x' + str(i)] = Dense(3, activation = 'softmax', kernel_regularizer = L1L2(l1 = 0., l2 = 0.), name = 'month_' + str(i))(x)

    xx1 = concatenate(list(prepare_list.values()))
    xx2 = Lambda(lambda x: x[:, 1::3], name = 'ranking_cause')(xx1)
    xx3 = Lambda(lambda x: x[:, 2::3], name = 'ranking_other')(xx1)

    losses = {}
    loss_weights = {}
    for i in range(time_length):
        losses['month_' + str(i)] = logloss(lambda3)
        loss_weights['month_' + str(i)] = 1

    losses['ranking_cause'] = rankingloss_cause
    losses['ranking_other'] = rankingloss_other
    loss_weights['ranking_cause'] = lambda4
    loss_weights['ranking_other'] = lambda4

    model = Model(input_tensor, list(prepare_list.values()) + [xx2] + [xx3])
    # model = Model(input_tensor, list(prepare_list.values()) + [xx2])
    model.compile(optimizer = Adam(lr),
                  loss = losses, # 'categorical_crossentropy',
                  loss_weights = loss_weights)
    model.fit(X_tr, Y_tr, epochs = 100, validation_data=(x_test, y_test_status_f), 
              batch_size = batch_size, shuffle = True, verbose = 0,
              callbacks = [
              ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0),
              EarlyStopping(patience = 5)])

    y_test_status_pred = model.predict(x_test)
    pred = np.array(y_test_status_pred[0:time_length])
    pred_dead = pred[:, :, 1]

    # c_index = concordance_index(durations_test//time_interval, 1 - pred_dead[-1, :], event_observed = events_test)
    # ibs = integrated_brier_score((y_test == 1).astype(int), pred_dead.T, np.arange(1, np.max(durations_test//time_interval)), np.max(durations_test//time_interval + 1))
    cif1 = pd.DataFrame(pred_dead, np.arange(time_length) + 1)
    ev1 = EvalSurv(1-cif1, durations_test//time_interval, events_test == 1, censor_surv='km')
    c_index = ev1.concordance_td('antolini')
    ibs = ev1.integrated_brier_score(np.arange(time_length)) # np.linspace(0, time_length, time_length + 1))

   

    c_indexes.append(c_index)
    ibss.append(ibs)
    print('C-index: {:.4f}'.format(c_index))
    print('IBS: {:.4f}'.format(ibs))


pd.DataFrame(data = {"cindex": c_indexes, "ibs": ibss}).to_csv("./data/results.ci.ssmtlr.csv", index=False)

# Compute the 95% confidence interval: conf_int
mean_cindex = np.mean(c_indexes)
mean_ibs = np.mean(ibss)

# Print the mean
print('mean cindex =', mean_cindex)
#### 0.7074
print('mean ibs =', mean_ibs)
#### 0.1014


ci_cindex = np.percentile(c_indexes, [2.5, 97.5])
ci_ibs = np.percentile(ibss, [2.5, 97.5])
 
# Print the confidence interval
print('confidence interval =', ci_cindex)
#### [0.7062 0.7087]
print('confidence interval =', ci_ibs)
#### [0.1005 0.1031]



