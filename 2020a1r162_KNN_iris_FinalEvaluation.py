#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[6]:


import numpy as np

from sklearn import datasets
from sklearn import neighbors

import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# In[7]:


iris = datasets.load_iris()
print(iris.keys())


# In[8]:


n_samples, n_features = iris.data.shape

print((n_samples, n_features))


# In[9]:


print(iris.data[0])


# In[10]:


print(iris.target.shape)


# In[11]:


print(iris.target)
print(iris.target_names)


# In[13]:


X, y = iris.data, iris.target

clf = neighbors.KNeighborsClassifier(n_neighbors=5)

clf.fit(X, y)


# In[14]:


result = clf.predict([[3, 5, 4, 2],])

print(iris.target_names[result])


# In[15]:


clf.predict_proba([[3, 5, 4, 2],])


# In[ ]:




