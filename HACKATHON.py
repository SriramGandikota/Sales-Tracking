
# coding: utf-8

# In[39]:


#Importing required libraries
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# In[2]:


#required functions for convertign strings to required data 
#For example converting the string of date time to timestamp in second
def convert_to_sec(l):
    l = l.split(':')
    t = int(l[2])
    t += int(l[1])*60
    t += int(l[0])*3600
    return t
def get_time_difference(x,y):
    x = x.split(' ')
    y = y.split(' ')
    X = time.mktime(datetime.datetime.strptime(x[0], "%Y-%m-%d").timetuple())
    Y = time.mktime(datetime.datetime.strptime(y[0], "%Y-%m-%d").timetuple())
    a = (convert_to_sec(x[1]) - convert_to_sec(y[1]))
    return (X-Y + a)


# In[3]:


#Loading the required data present across three input files
impression_info = pd.read_csv('train.csv')
user_info = pd.read_csv('view_log.csv')
item_info = pd.read_csv('item_data.csv')


# # Exploratory Data Analysis
# 

# In[4]:


#impression information after sorting
impression_info.sort_values(by =['user_id','impression_time'],inplace=True)
impression_info.head()


# In[5]:


#view log information of the user after sorting
user_info.sort_values(by =['user_id','server_time'],inplace=True)
user_info.head()


# In[6]:


#item information
item_info.head()


# In[7]:


# since the preprocessing takes a lot of time we have files that have preprocessed data


# In[9]:


#Size of the data
print('Impression Info {} \nUser Info {} \nItem Info {}'.format(impression_info.shape,user_info.shape,item_info.shape) )


# ## Exploration of the Data

# As we saw that the data needs to be merged together to make sense of it. Hence we have followed a procedure that has been exaplained in the report in detailed.

# In[11]:


#Number of users in the impression info whose information is not in user info (if any)
cnt = 0
for user in impression_info['user_id']:
    if(user not in user_info['user_id']):
        cnt+=1
print("Number of Users whose information is missing = {}\n".format(cnt))


# In[ ]:


# NO NEED TO RUN THIS CELL TO TEST THE MODEL
# This cell contains the implementation of the function that is extracting an important new feature that is
# described in the report as "Time duration between log time and impression time".
ind1 = 0
log_time = []
is_new = []
curr_item = []
s = "-1"
prev_user = -1
prev_item = -1
for ind in range(0,impression_info.shape[0]):
    if(ind1==user_info.shape[0]):
        break
    if(prev_user!=impression_info['user_id'].values[ind]):
        s="-1"
        prev_user = -1
        prev_item=  -1
    while(impression_info['user_id'].values[ind]>user_info['user_id'].values[ind1]):
        ind1+=1
    while(get_time_difference(impression_info['impression_time'].values[ind],user_info['server_time'].values[ind1])>=0
         and impression_info['user_id'].values[ind]==user_info['user_id'].values[ind1]):
        s = user_info['server_time'].values[ind1]
        prev_user = user_info['user_id'].values[ind1]
        prev_item = user_info['item_id'].values[ind1]
        ind1+=1
    if(s=="-1"):
        log_time.append(-1)
        is_new.append(1)
        curr_item.append(prev_item)
    else:
        is_new.append(0)
        log_time.append(get_time_difference(impression_info['impression_time'].values[ind],s))
        curr_item.append(prev_item)
        ind1-=1


# In[ ]:


# NO NEED TO RUN THIS CELL TO TEST THE MODEL
# Adding the newly calculated features to the data frame.
data = impression_info
data['log_time'] = log_time
data['item_id'] = curr_item


# In[ ]:


# NO NEED TO RUN THIS CELL TO TEST THE MODEL
#merging the data frames based on items that the given user might have seen
#in the most recent log
data = pd.merge(data,item_info,on='item_id')


# In[ ]:


# NO NEED TO RUN THIS CELL TO TEST THE MODEL
# One of the columns in the data has string values and hence we are mapping them to integer values.
data1['os_version'] = data1['os_version'].map({'old':1,'latest':2,'intermediate':3}).astype(int)
#Dropping columns that do not add any information to the data 
data1 = data.drop(columns = ['impression_id','impression_time']) 
#Saving the cumulative data into a new file
data1.to_csv('Further_Prepreocessed_data.csv')
# We split the 'Further_Prepreocessed_data.csv' file into two seperate files with only positive and only negative
# examples for a better understanding.


# In[41]:


# Reading positive and negative examples.
data0 = pd.read_csv('negatives.csv')
data1 = pd.read_csv('positives.csv')


# In[42]:


# Printing the number of positive and negative examples
print("Negative examples = ",data0.shape,"\nPositive examples = ",data1.shape)
# We can see that the data is very skewed as only 5% of the data are positive examples and the rest are negative examples


# In[43]:


# One of the columns in the data has string values and hence we are mapping them to integer values.
data0['os_version'] = data0['os_version'].map({'old':1,'latest':2,'intermediate':3}).astype(int)
data1['os_version'] = data1['os_version'].map({'old':1,'latest':2,'intermediate':3}).astype(int)


# In[44]:


# Normalizing the data
for i in data0.columns:
    if(i != 'is_click'):
        data0_norm[i] = (data0[i]-data0[i].mean())/data0[i].std()
        data1_norm[i] = (data1[i]-data1[i].mean())/data1[i].std()
    else:
        data0_norm[i] = data0[i]
        data1_norm[i] = data1[i]


# In[45]:


data0_norm.drop(columns=data0_norm.columns[:3],inplace=True)
data1_norm.drop(columns=data1_norm.columns[:3],inplace=True)


# In[57]:


data = []
j = 0
i = 0
while(i<=data0.shape[0] and j<20):
    data.append(data0[i:i+10000])
    i += 10000
    j += 1


# In[58]:


len(data)


# # Model Training and Building

# In[60]:


# Building multiple models for prediction.
# We tried running the models on normalized data but the resluts were not as good as un-normalized data. This is
# probably because many columns have categorical data.
from sklearn.decomposition import PCA 
from xgboost import XGBClassifier
pca = PCA(n_components = 7) 
  
p = []
overall = 0
for i in range(0,20):
    train = pd.concat([data1,data[i]])
    Y = train['is_click']
    del train['is_click'] 
    train_x,test_x,train_y,test_y = train_test_split(train,Y,test_size=0.1,random_state=67)
    #pca
    train_x = pca.fit_transform(train_x) 
    test_x = pca.transform(test_x)
    model1 = LogisticRegression()    
    model2 = RandomForestClassifier()    
    model3 = XGBClassifier()
    model4 = VotingClassifier(estimators=[('lr',model1),('ran',model2),('xgb',model3)],voting='soft')

    model4.fit(train_x,train_y)
    overall += accuracy_score(model4.predict(test_x),test_y)
print(overall/20.0)


# In[ ]:


#Loading Model
loaded_model = pickle_load(open('Without_PCA.sav','rb'))

