#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pip install xlrd


# In[3]:


df=pd.read_excel("cluster_april.xlsx")


# In[4]:


df


# In[5]:


pip install -U scikit-learn


# In[6]:


import sklearn


# In[7]:


from sklearn.cluster import KMeans


# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[9]:


pip install matplotlib


# In[10]:


pip install numpy


# In[12]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df


# ## Plotting on scatter plot the 2 features of each test

# In[16]:


plt.scatter(df['Frequency'],df['Avg_revenue'])


# ## Scaling the data using minmaxscaler

# In[25]:


scaler = MinMaxScaler()
df[['Frequency']]=scaler.fit_transform(df[['Frequency']])
df[['Avg_revenue']]=scaler.fit_transform(df[['Avg_revenue']])
df


# In[26]:


plt.scatter(df['Frequency'],df['Avg_revenue'])


# ## Finding optimum Value of k-number of clusters

# In[28]:


k_rng= range(1,11)
sse=[]
for k in k_rng:
    km= KMeans(n_clusters=k)
    km.fit(df[['Frequency','Avg_revenue']])
    sse.append(km.inertia_)


# In[29]:


sse


# In[30]:


plt.plot(k_rng,sse)
plt.xlabel('k')
plt.ylabel('Sum squared error')


# ## K= 5 is selected as optimum value of cluster numbers using elow-joint analysis

# In[31]:


km = KMeans(n_clusters=5)
km


# In[33]:


y_predicted= km.fit_predict(df[['Frequency','Avg_revenue']])
y_predicted


# In[34]:


df['cluster']=y_predicted
df


# ## plotting these 5 clusters after segregating them

# In[40]:


df0= df[df['cluster']==0]
df1= df[df['cluster']==1]
df2= df[df['cluster']==2]
df3= df[df['cluster']==3]
df4= df[df['cluster']==4]

df3


# In[41]:


df0


# In[42]:


df1


# In[44]:


df4


# In[51]:


plt.scatter(df0['Frequency'],df0['Avg_revenue'],color='yellow')
plt.scatter(df1['Frequency'],df1['Avg_revenue'],color='green')
plt.scatter(df2['Frequency'],df2['Avg_revenue'],color='black')
plt.scatter(df3['Frequency'],df3['Avg_revenue'],color='red')
plt.scatter(df4['Frequency'],df4['Avg_revenue'],color='blue')
plt.xlabel('Frequency')
plt.ylabel('Avg-price')
fig= plt.figure(figsize=(20,15))
plt.show()


# In[ ]:




