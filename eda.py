#!/usr/bin/env python
# coding: utf-8

# # eda
# 
# Use the "Run" button to execute the code.

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


# Execute this to save new versions of the notebook
jovian.commit(project="eda")


# In[21]:


import pandas as pd


# In[3]:


dataset=pd.read_csv("food_coded.csv")


# In[4]:


dataset.head()


# In[5]:


dataset.tail()


# In[14]:


dataset.columns


# In[28]:


column=['cook','eating_out','employment','ethnic_food', 'exercise','fruit_day','income','on_off_campus','pay_meal_out','sports','veggies_day','indian_food','healthy_meal']


# In[29]:


data=dataset[column]


# In[19]:


data


# In[30]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
ax=data.boxplot(figsize=(16,6))
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)


# In[33]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


km=KMeans(n_clusters=3)
km


# In[38]:


y_predicted=km.fit_predict(dataset[['income','calories_chicken']])
y_predicted


# In[ ]:




