#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Read data
# 

# In[2]:


data=pd.read_csv(r"C:\Users\hp\Desktop\Cvoid-19\Pdata.csv")
data.head(10)


# In[3]:


data.tail(10)


# In[4]:


data.info()


# In[5]:


data['Fever'].value_counts()


# In[6]:


data['BodyPain'].value_counts()


# In[7]:


data['Age'].value_counts()


# In[8]:


data['RunnyNose'].value_counts()


# In[9]:


data['Difficulty In breathing'].value_counts()


# In[10]:


data['Dry Cough'].value_counts()


# In[11]:


data.describe()


# # # Train Test Spliting

# In[12]:


import numpy as np


# In[13]:


def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)* ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[14]:


np.random.permutation(7)


# In[15]:


train, test = data_split(data, 0.2)
data.to_numpy()


# In[16]:


train


# In[17]:


test


# In[18]:


X_train = train[['Fever','BodyPain','Age','RunnyNose','Dry Cough','Difficulty In breathing']].to_numpy()
X_test = test[['Fever','BodyPain','Age','RunnyNose','Dry Cough','Difficulty In breathing']].to_numpy()


# In[19]:


Y_train = train[['Probability of Covoid19']].to_numpy().reshape(2008,)
Y_test = test[['Probability of Covoid19']].to_numpy().reshape(501,)


# In[20]:


Y_train


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)


# In[23]:


inputFeatures = [102, 1, 35, 1, 1, 1]
infProb = clf.predict_proba([inputFeatures])[0][1]


# In[24]:


infProb


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




