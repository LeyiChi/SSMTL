#!/usr/bin/env python
# coding: utf-8

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


np.random.seed(1234)
tf.random.set_seed(2021)



df_train = pd.read_csv("../data/007-data_crs_train_py.csv")
df_test = pd.read_csv("../data/007-data_crs_test_py.csv")



cols_standardize = ['age', "positivelymph"]
cols_leave = [x for x in df_train.columns.to_list() if x not in ["time", "crstatus", "age", "positivelymph"]]

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')



def get_y_labels(status, time):
    ret = np.zeros((status.shape[0], np.max(time) + 1))
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



y_train_status = to_categorical(y_train)
y_test_status = to_categorical(y_test)


def reshape_y(y):
    dim = y.shape[1]
    ret = []
    for i in range(dim):
        ret.append(y[:, i, 0:3])
    return ret        


y_train_status = reshape_y(y_train_status)
y_test_status = reshape_y(y_test_status)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

y_train_status_f = y_train_status + [y_train] + [y_train]
y_test_status_f = y_test_status + [y_test] + [y_test]


get_target = lambda df: (df['time'].values, df['crstatus'].values)
durations_train, events_train = get_target(df_train)
durations_test, events_test = get_target(df_test)


def logloss(lambda3):
    def loss(y_true, y_pred):
        mask_alive = y_true[:, 0]
        mask_dead_cause = y_true[:, 1]
        mask_dead_other = y_true[:, 2]
        mask_censored = 1 - (mask_alive + mask_dead_cause + mask_dead_other)
        logloss = -1 * k.mean(mask_alive * k.log(y_pred[:, 0]) + mask_dead_cause * k.log(y_pred[:, 1])
                         + mask_dead_other * k.log(y_pred[:, 2]))
        - lambda3 * k.mean(y_pred[:, 1] * mask_censored * k.log(y_pred[:, 1]) + y_pred[:, 0] * mask_censored * k.log(y_pred[:, 0])
                   + y_pred[:, 2] * mask_censored * k.log(y_pred[:, 2]))
        return logloss
    return loss


def rankingloss_cause(y_true, y_pred, name = None):
    ranking_loss = 0
    for i in range(time_length):
        for j in range(i + 1, time_length, 1):
            tmp = y_pred[:, i] - y_pred[:, j]
            tmp1 = tmp > 0
            tmp1 = tf.cast(tmp1, tf.float32)
            ranking_loss = ranking_loss + k.mean((tmp1 * tmp * (j - i)))
    return ranking_loss


def rankingloss_other(y_true, y_pred, name = None):
    ranking_loss = 0
    for i in range(time_length):
        for j in range(i + 1, time_length, 1):
            tmp = y_pred[:, i] - y_pred[:, j]
            tmp1 = tmp > 0
            tmp1 = tf.cast(tmp1, tf.float32)
            ranking_loss = ranking_loss + k.mean((tmp1 * tmp * (j - i)))
    return ranking_loss


ssmtlr_cv_results = pd.read_csv("../data/cv.results.ssmtlr.csv")
print(ssmtlr_cv_results["cindex"].values.max())
ind_best = ssmtlr_cv_results["cindex"].values.argmax()

lambda3 = ssmtlr_cv_results.iloc[ind_best, 0]
lambda4 = ssmtlr_cv_results.iloc[ind_best, 1]
lr = ssmtlr_cv_results.iloc[ind_best, 2]
batch_size = ssmtlr_cv_results.iloc[ind_best, 3]



input_tensor = Input((x_train.shape[1],))
x = input_tensor

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
model.compile(optimizer = Adam(lr),
              loss = losses,
              loss_weights = loss_weights)


model.fit(x_train, y_train_status_f, epochs = 100, validation_data=(x_test, y_test_status_f), 
          batch_size = batch_size, shuffle = True, verbose = 1,
          callbacks = [
          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0),
          EarlyStopping(patience = 5)])


y_test_status_pred = model.predict(x_test)
pred = np.array(y_test_status_pred[0:time_length])
pred_dead = pred[:, :, 1]



cif1 = pd.DataFrame(pred_dead, np.arange(time_length) + 1)
ev1 = EvalSurv(1-cif1, durations_test//time_interval, events_test == 1, censor_surv='km')
c_index = ev1.concordance_td('antolini')
ibs = ev1.integrated_brier_score(np.arange(time_length))


print('C-index: {:.4f}'.format(c_index))
print('IBS: {:.4f}'.format(ibs))




# ------------------------------


testDf = pd.DataFrame(x_test)
testNo = testDf.shape[0]
testCol = testDf.shape[1]


vimp = []
for i in range(testCol):
    print(i)
    errors = []
    for count in range(100):
        testTmp = testDf.copy()
        if i == 0 or i == 1:
            sigma = np.std(testDf[i])
            epsilon = 5
            testTmp[i] = testTmp[i] + np.random.normal(0, epsilon * sigma, testNo)
        else:
            s = np.random.binomial(1, 0.5, testNo)
            testTmp[i] = testTmp[i] * (1 - s) + (1 - testTmp[i]) * s 
        Y_test_pred_tmp = model.predict(np.asarray(testTmp))
        pred_tmp = np.array(Y_test_pred_tmp[0:time_length])
        pred_dead_tmp = pred_tmp[:, :, 1]
        cif1 = pd.DataFrame(pred_dead_tmp, np.arange(time_length) + 1)
        ev1 = EvalSurv(1-cif1, durations_test//time_interval, events_test == 1, censor_surv='km')
        c_index = ev1.concordance_td('antolini')
        errors.append(c_index)
    vimp.append(np.mean(errors))


result = pd.DataFrame({"vimp": vimp})
result.to_csv('../data/ssmtlr_vimp.csv', index = False)


