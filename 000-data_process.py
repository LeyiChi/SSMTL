#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### 1 - Importing packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
#%pylab inline


# In[2]:


train = pd.read_csv("./data/007-data_crs_train.csv")
test = pd.read_csv("./data/007-data_crs_test.csv")
train["flag"] = 1
test["flag"] = 0


# In[3]:


train.shape


# In[4]:


test.shape


# In[5]:


data = pd.concat([train, test])
data["os"] = (data["os"] == 4).astype(int)


# In[6]:

features_cat = ["race", "site", "hist", "grade", "ajcc7t", "ajcc7n", "ajcc7m", "surgery", "radiation"]
features_con = ["age", "positivelymph"]


# In[7]:


df_dummy = pd.get_dummies(data[features_cat])
data = pd.concat([data, df_dummy], axis = 1)


# In[8]:


train = data[data["flag"] == 1]
test = data[data["flag"] == 0]


# In[9]:


features = df_dummy.columns.to_list() + features_con


# In[10]:


len(features)


# In[11]:


train_sel = train[["time", "crstatus"] + features]
test_sel = test[["time", "crstatus"] + features]


# In[12]:


train_sel.head(100)


# In[13]:


train_sel.to_csv("./data/007-data_crs_train_py.csv", index = False)
test_sel.to_csv("./data/007-data_crs_test_py.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:




