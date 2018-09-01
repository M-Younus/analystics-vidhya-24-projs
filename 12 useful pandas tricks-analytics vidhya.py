
# coding: utf-8

# https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/

# In[2]:


import pandas as pd
import numpy as np
data=pd.read_csv("train.csv",index_col="Loan_ID")


# In[3]:


data.head()


# In[4]:


data[(data['Gender']=='Female') & (data['Education']=='Not Graduate') & (data['Loan_Status']=='Y')]


# In[5]:


data.apply(lambda x:sum(x.isnull()),axis=0)


# In[6]:


data.apply(lambda x:sum(x.isnull()),axis=1).head()


# In[7]:


impute_grps=data.pivot_table(values=['LoanAmount'],index=['Gender','Married','Self_Employed'],aggfunc=np.mean)
impute_grps


# In[9]:


pd.crosstab(data['Credit_History'],data['Loan_Status'],margins=True)


# In[11]:


percConvert=lambda ser:ser/float(ser[-1])
pd.crosstab(data['Credit_History'],data['Loan_Status'],margins=True).apply(percConvert,axis=1)


# In[18]:


df = pd.DataFrame({
   'col1' : ['A', 'A', 'B', 'D', 'D', 'B'],
     'col2' : [2, 1, 9, 0, 7, 4],
    'col3': [0, 1, 9, 4, 2, 3],
 })


# In[19]:


df


# In[20]:


df.sort_values(by=['col1', 'col2'])


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data.boxplot(column='ApplicantIncome',by='Loan_Status')


# In[28]:


data.hist(column='ApplicantIncome',by='Loan_Status',bins=30)


# In[30]:


pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3, labels=["good","medium","bad"])

