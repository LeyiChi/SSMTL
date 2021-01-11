#!/usr/bin/env python
# coding: utf-8


#### 1 - Importing packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


train = pd.read_csv("../data/007-data_crs_train.csv")
test = pd.read_csv("../data/007-data_crs_test.csv")
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
train_sel = train[["time", "crstatus"] + features]
test_sel = test[["time", "crstatus"] + features]
train_sel.to_csv("../data/007-data_crs_train_py.csv", index = False)
test_sel.to_csv("../data/007-data_crs_test_py.csv", index = False)

