#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import h5py
import pickle
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

alldata = pd.concat([df_train, df_test])

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
ibs = ev1.integrated_brier_score(np.arange(time_length)) # np.linspace(0, time_length, time_length + 1))

print('C-index: {:.4f}'.format(c_index))
print('IBS: {:.4f}'.format(ibs))

alldata = pd.concat([pd.DataFrame(x_train), pd.DataFrame(x_test)])
recordNum = alldata.shape[0]
dim = alldata.shape[1]


survivalByVar = {'age': pd.DataFrame(columns = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
                        'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26']),
                 'positivelymph': pd.DataFrame(columns = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10',
                        'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26']),
                 'race': pd.DataFrame(columns = ['White', 'Black', 'Others']),
                 'site': pd.DataFrame(columns = ['Left colon', 'Right colon', 'Rectum']),
                 'histology': pd.DataFrame(columns = ['Adenocarcinoma', 'Others']),
                 'grade': pd.DataFrame(columns = ['I', 'II', 'III', 'IV']),
                 'ajcc7t': pd.DataFrame(columns = ['T1', 'T2', 'T3', 'T4a', 'T4b']),
                 'ajcc7n': pd.DataFrame(columns = ['N0', 'N1a', 'N1b', 'N2a', 'N2b']),
                 'ajcc7m': pd.DataFrame(columns = ['M0', 'M1']),
                 'surgery': pd.DataFrame(columns = ['No surgery', 'Local', 'Partial', 'Subtotal', 'Total', 'Others']),
                 'radiation': pd.DataFrame(columns = ['No need', 'Need'])}


survivalByAge = []


# age
for i in np.linspace(np.min(alldata[0]), np.max(alldata[0]), 26):
    alldataTmp = alldata.copy()
    alldataTmp[0] = i
    alldata_pred = model.predict(np.asarray(alldataTmp))
    pred = np.array(alldata_pred[0:time_length])
    pred_dead = pred[:, :, 1]
    survivalByAge.append(pred_dead)


with open('./survivalByAge.pk', 'wb') as f:
    pickle.dump(survivalByAge, f)

partial_dep = np.mean(survivalByAge, axis = 2)
par_dep_5_age = partial_dep[:, 9]
par_dep_4_age = partial_dep[:, 7]
par_dep_3_age = partial_dep[:, 5]
par_dep_2_age = partial_dep[:, 3]
par_dep_1_age = partial_dep[:, 1]

age = np.linspace(0, 1, 26)
age = np.round(age * (107 - 11) + 11)
nonlinear_age = pd.DataFrame({'age': age, '1-year': par_dep_1_age, '2-year': par_dep_2_age,
                             '3-year': par_dep_3_age, '4-year': par_dep_4_age, '5-year': par_dep_5_age})
nonlinear_age.to_csv('../results/nonlinear_age.csv', index = False)



# PLN
survivalByPLN = []
for i in np.linspace(np.min(alldata[0]), np.max(alldata[0]), 26):
    alldataTmp = alldata.copy()
    alldataTmp[1] = i
    alldata_pred = model.predict(np.asarray(alldataTmp))
    pred = np.array(alldata_pred[0:time_length])
    pred_dead = pred[:, :, 1]
    survivalByPLN.append(pred_dead)
with open('./survivalByPLN.pk', 'wb') as f:
    pickle.dump(survivalByPLN, f)

partial_dep_pln = np.mean(survivalByPLN, axis = 2)
par_dep_5_pln = partial_dep_pln[:, 9]
par_dep_4_pln = partial_dep_pln[:, 7]
par_dep_3_pln = partial_dep_pln[:, 5]
par_dep_2_pln = partial_dep_pln[:, 3]
par_dep_1_pln = partial_dep_pln[:, 1]

pln = np.linspace(0, 1, 26)
pln = np.round(pln * (np.max(alldata['positivelymph']) - 0) + 0)
nonlinear_pln = pd.DataFrame({'age': pln, '1-year': par_dep_1_pln, '2-year': par_dep_2_pln,
  '3-year': par_dep_3_pln, '4-year': par_dep_4_pln, '5-year': par_dep_5_pln})
nonlinear_pln.to_csv('../results/nonlinear_pln.csv', index = False)


# Race
survivalByRace = []
alldataTmp = alldata.copy()
alldataTmp[2] = 1
alldataTmp[3] = 0
alldataTmp[4] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByRace.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[2] = 0
alldataTmp[3] = 1
alldataTmp[4] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByRace.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[2] = 0
alldataTmp[3] = 0
alldataTmp[4] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByRace.append(pred_dead)

with open('./survivalByRace.pk', 'wb') as f:
    pickle.dump(survivalByRace, f)

pd.DataFrame(survivalByRace[0].T).to_csv('./results/nonlinear_race_white.csv', index = False)
pd.DataFrame(survivalByRace[1].T).to_csv('./results/nonlinear_race_black.csv', index = False)
pd.DataFrame(survivalByRace[2].T).to_csv('./results/nonlinear_race_others.csv', index = False)
partial_dep_race = np.mean(survivalByRace, axis = 2)
pd.DataFrame(partial_dep_race).to_csv('./results/nonlinear_race.csv', index = False)



# Site
survivalBySite = []

alldataTmp = alldata.copy()
alldataTmp[5] = 1
alldataTmp[6] = 0
alldataTmp[7] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySite.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[5] = 0
alldataTmp[6] = 1
alldataTmp[7] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySite.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[5] = 0
alldataTmp[6] = 0
alldataTmp[7] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySite.append(pred_dead)

with open('./survivalBySite.pk', 'wb') as f:
    pickle.dump(survivalBySite, f)

pd.DataFrame(survivalBySite[0].T).to_csv('./results/nonlinear_site_left_colon.csv', index = False)
pd.DataFrame(survivalBySite[1].T).to_csv('./results/nonlinear_site_right_colon.csv', index = False)
pd.DataFrame(survivalBySite[2].T).to_csv('./results/nonlinear_site_rectum.csv', index = False)
partial_dep_site = np.mean(survivalBySite, axis = 2)
pd.DataFrame(partial_dep_site).to_csv('./results/nonlinear_site.csv', index = False)


# hist
survivalByHist = []

alldataTmp = alldata.copy()
alldataTmp[8] = 1
alldataTmp[9] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByHist.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[8] = 0
alldataTmp[9] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByHist.append(pred_dead)

with open('./survivalByHist.pk', 'wb') as f:
    pickle.dump(survivalByHist, f)

pd.DataFrame(survivalByHist[0].T).to_csv('./results/nonlinear_hist_adeno.csv', index = False)
pd.DataFrame(survivalByHist[1].T).to_csv('./results/nonlinear_hist_others.csv', index = False)
partial_dep_hist = np.mean(survivalByHist, axis = 2)
pd.DataFrame(partial_dep_hist).to_csv('./results/nonlinear_hist.csv', index = False)


# grade
survivalByGrade = []

alldataTmp = alldata.copy()
alldataTmp[10] = 1
alldataTmp[11] = 0
alldataTmp[12] = 0
alldataTmp[13] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByGrade.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[10] = 0
alldataTmp[11] = 1
alldataTmp[12] = 0
alldataTmp[13] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByGrade.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[10] = 0
alldataTmp[11] = 0
alldataTmp[12] = 1
alldataTmp[13] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByGrade.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[10] = 0
alldataTmp[11] = 0
alldataTmp[12] = 0
alldataTmp[13] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByGrade.append(pred_dead)

with open('./survivalByGrade.pk', 'wb') as f:
    pickle.dump(survivalByGrade, f)

pd.DataFrame(survivalByGrade[0].T).to_csv('./results/nonlinear_grade_1.csv', index = False)
pd.DataFrame(survivalByGrade[1].T).to_csv('./results/nonlinear_grade_2.csv', index = False)
pd.DataFrame(survivalByGrade[2].T).to_csv('./results/nonlinear_grade_3.csv', index = False)
pd.DataFrame(survivalByGrade[3].T).to_csv('./results/nonlinear_grade_4.csv', index = False)
partial_dep_grade = np.mean(survivalByGrade, axis = 2)
pd.DataFrame(partial_dep_grade).to_csv('./results/nonlinear_grade.csv', index = False)


# AJCC7T
survivalByAJCC7T = []

alldataTmp = alldata.copy()
alldataTmp[14] = 1
alldataTmp[15] = 0
alldataTmp[16] = 0
alldataTmp[17] = 0
alldataTmp[18] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7T.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[14] = 0
alldataTmp[15] = 1
alldataTmp[16] = 0
alldataTmp[17] = 0
alldataTmp[18] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7T.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[14] = 0
alldataTmp[15] = 0
alldataTmp[16] = 1
alldataTmp[17] = 0
alldataTmp[18] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7T.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[14] = 0
alldataTmp[15] = 0
alldataTmp[16] = 0
alldataTmp[17] = 1
alldataTmp[18] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7T.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[14] = 0
alldataTmp[15] = 0
alldataTmp[16] = 0
alldataTmp[17] = 0
alldataTmp[18] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7T.append(pred_dead)

with open('./survivalByAJCC7T.pk', 'wb') as f:
    pickle.dump(survivalByAJCC7T, f)


pd.DataFrame(survivalByAJCC7T[0].T).to_csv('./results/nonlinear_Tstage_T1.csv', index = False)
pd.DataFrame(survivalByAJCC7T[1].T).to_csv('./results/nonlinear_Tstage_T2.csv', index = False)
pd.DataFrame(survivalByAJCC7T[2].T).to_csv('./results/nonlinear_Tstage_T3.csv', index = False)
pd.DataFrame(survivalByAJCC7T[3].T).to_csv('./results/nonlinear_Tstage_T4a.csv', index = False)
pd.DataFrame(survivalByAJCC7T[4].T).to_csv('./results/nonlinear_Tstage_T4b.csv', index = False)
partial_dep_AJCC7T = np.mean(survivalByAJCC7T, axis = 2)
pd.DataFrame(partial_dep_AJCC7T).to_csv('./results/nonlinear_Tstage.csv', index = False)


# AJCC7N
survivalByAJCC7N = []

alldataTmp = alldata.copy()
alldataTmp[19] = 1
alldataTmp[20] = 0
alldataTmp[21] = 0
alldataTmp[22] = 0
alldataTmp[23] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7N.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[19] = 0
alldataTmp[20] = 1
alldataTmp[21] = 0
alldataTmp[22] = 0
alldataTmp[23] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7N.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[19] = 0
alldataTmp[20] = 0
alldataTmp[21] = 1
alldataTmp[22] = 0
alldataTmp[23] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7N.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[19] = 0
alldataTmp[20] = 0
alldataTmp[21] = 0
alldataTmp[22] = 1
alldataTmp[23] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7N.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[19] = 0
alldataTmp[20] = 0
alldataTmp[21] = 0
alldataTmp[22] = 0
alldataTmp[23] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7N.append(pred_dead)

with open('./survivalByAJCC7N.pk', 'wb') as f:
    pickle.dump(survivalByAJCC7N, f)

pd.DataFrame(survivalByAJCC7N[0].T).to_csv('./results/nonlinear_AJCC7N_1.csv', index = False)
pd.DataFrame(survivalByAJCC7N[1].T).to_csv('./results/nonlinear_AJCC7N_2.csv', index = False)
pd.DataFrame(survivalByAJCC7N[2].T).to_csv('./results/nonlinear_AJCC7N_3.csv', index = False)
pd.DataFrame(survivalByAJCC7N[3].T).to_csv('./results/nonlinear_AJCC7N_4.csv', index = False)
pd.DataFrame(survivalByAJCC7N[4].T).to_csv('./results/nonlinear_AJCC7N_5.csv', index = False)
partial_dep_AJCC7N = np.mean(survivalByAJCC7N, axis = 2)
pd.DataFrame(partial_dep_AJCC7N).to_csv('./results/nonlinear_Nstage.csv', index = False)


# AJCC7M
survivalByAJCC7M = []

alldataTmp = alldata.copy()
alldataTmp[24] = 1
alldataTmp[25] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7M.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[24] = 0
alldataTmp[25] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByAJCC7M.append(pred_dead)

with open('./survivalByAJCC7M.pk', 'wb') as f:
    pickle.dump(survivalByAJCC7M, f)

pd.DataFrame(survivalByAJCC7M[0].T).to_csv('./results/nonlinear_AJCC7M_1.csv', index = False)
pd.DataFrame(survivalByAJCC7M[1].T).to_csv('./results/nonlinear_AJCC7M_2.csv', index = False)
partial_dep_AJCC7M = np.mean(survivalByAJCC7M, axis = 2)
pd.DataFrame(partial_dep_AJCC7M).to_csv('./results/nonlinear_AJCC7M.csv', index = False)


# Surgery
survivalBySurgery = []

alldataTmp = alldata.copy()
alldataTmp[26] = 1
alldataTmp[27] = 0
alldataTmp[28] = 0
alldataTmp[29] = 0
alldataTmp[30] = 0
alldataTmp[31] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySurgery.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[26] = 0
alldataTmp[27] = 1
alldataTmp[28] = 0
alldataTmp[29] = 0
alldataTmp[30] = 0
alldataTmp[31] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySurgery.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[26] = 0
alldataTmp[27] = 0
alldataTmp[28] = 1
alldataTmp[29] = 0
alldataTmp[30] = 0
alldataTmp[31] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySurgery.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[26] = 0
alldataTmp[27] = 0
alldataTmp[28] = 0
alldataTmp[29] = 1
alldataTmp[30] = 0
alldataTmp[31] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySurgery.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[26] = 0
alldataTmp[27] = 0
alldataTmp[28] = 0
alldataTmp[29] = 0
alldataTmp[30] = 1
alldataTmp[31] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySurgery.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[26] = 0
alldataTmp[27] = 0
alldataTmp[28] = 0
alldataTmp[29] = 0
alldataTmp[30] = 0
alldataTmp[31] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalBySurgery.append(pred_dead)

with open('./survivalBySurgery.pk', 'wb') as f:
    pickle.dump(survivalBySurgery, f)

pd.DataFrame(survivalBySurgery[0].T).to_csv('./results/nonlinear_Surgery_1.csv', index = False)
pd.DataFrame(survivalBySurgery[1].T).to_csv('./results/nonlinear_Surgery_2.csv', index = False)
pd.DataFrame(survivalBySurgery[2].T).to_csv('./results/nonlinear_Surgery_3.csv', index = False)
pd.DataFrame(survivalBySurgery[3].T).to_csv('./results/nonlinear_Surgery_4.csv', index = False)
pd.DataFrame(survivalBySurgery[4].T).to_csv('./results/nonlinear_Surgery_5.csv', index = False)
pd.DataFrame(survivalBySurgery[5].T).to_csv('./results/nonlinear_Surgery_6.csv', index = False)
partial_dep_Surgery = np.mean(survivalBySurgery, axis = 2)
pd.DataFrame(partial_dep_Surgery).to_csv('./results/nonlinear_Surgery.csv', index = False)


# Radiation
survivalByRadiation = []

alldataTmp = alldata.copy()
alldataTmp[32] = 1
alldataTmp[33] = 0
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByRadiation.append(pred_dead)

alldataTmp = alldata.copy()
alldataTmp[32] = 0
alldataTmp[33] = 1
alldata_pred = model.predict(np.asarray(alldataTmp))
pred = np.array(alldata_pred[0:time_length])
pred_dead = pred[:, :, 1]
survivalByRadiation.append(pred_dead)

with open('./survivalByRadiation.pk', 'wb') as f:
    pickle.dump(survivalByRadiation, f)

pd.DataFrame(survivalByRadiation[0].T).to_csv('./results/nonlinear_radiation_1.csv', index = False)
pd.DataFrame(survivalByRadiation[1].T).to_csv('./results/nonlinear_radiation_2.csv', index = False)
partial_dep_radiation = np.mean(survivalByRadiation, axis = 2)
pd.DataFrame(partial_dep_radiation).to_csv('./results/nonlinear_radiation.csv', index = False)


