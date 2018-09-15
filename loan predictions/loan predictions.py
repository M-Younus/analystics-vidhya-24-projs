
# coding: utf-8

# https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('D:\DM\analystics vidhya 24 projs prac\loan predictions\train.csv')


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df['Property_Area'].value_counts()


# In[6]:


df['ApplicantIncome'].hist(bins=50)


# In[7]:


df.boxplot(column='ApplicantIncome')


# In[8]:


df.boxplot(column='ApplicantIncome',by='Education')


# In[9]:


df['LoanAmount'].hist(bins=50)


# In[10]:


df.boxplot(column='LoanAmount')


# In[11]:


temp1=df['Credit_History'].value_counts(ascending=True)
temp2=df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x:x.map({'Y':1,'N':0}).mean())


# In[12]:


print('Frequency Table for Credit History')
print(temp1)

print('\nProbability of getting loan for each Credit History class:abs')
print(temp2)


# In[13]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title('Applicants by Credit_History')
temp1.plot(kind='bar')

ax2=fig.add_subplot(122)
temp2.plot(kind='bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title('Probablility of getting loan by credit history')


# In[14]:


temp3=pd.crosstab([df['Credit_History']],[df['Loan_Status']])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[15]:


temp3=pd.crosstab([df['Credit_History'],df['Gender']],[df['Loan_Status']])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)


# In[16]:


df.apply(lambda x:sum(x.isnull()),axis=0)


# In[17]:


# df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# In[18]:


df.apply(lambda x:sum(x.isnull()),axis=0)


# In[19]:


df.boxplot(column='LoanAmount',by=['Education','Self_Employed'])


# In[20]:


df['Self_Employed'].value_counts()


# In[21]:


df['Self_Employed'].fillna('No',inplace=True)


# In[22]:


df['Self_Employed'].value_counts()


# In[23]:


table=df.pivot_table(values='LoanAmount',index='Self_Employed',columns='Education',aggfunc=np.median)

fage=lambda x:table.loc[x['Self_Employed'],x['Education']]

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage,axis=1),inplace=True)


# In[24]:


df.apply(lambda x:sum(x.isnull()),axis=0)


# In[25]:


df['LoanAmount'].hist(bins=20)


# In[26]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[27]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20)
# df['TotalIncome_log'].hist(bins=20)


# 5.Building a predictive Model in python

# In[29]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# In[30]:


from sklearn.preprocessing import LabelEncoder
var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le=LabelEncoder()
for i in var_mod:
    df[i]=le.fit_transform(df[i])
df.dtypes

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import metrics

#generic function for making a classification model and accessing performance,
def classification_model(model,data,predictors,outcome):
    #fit the model
    model.fit(data[predictors],data[outcome])
    
    #make predictions on training set
    predictions=model.predict(data[predictors])
    
    #print accuracy
    accuracy=metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))
    
    #perform k-fold cross-validation with 5 folds
    kf=KFold(data.shape[0],n_folds=5)
    error=[]
    for train,test in kf:
        #filter training data
        train_predictors=(data[predictors].iloc[train,:])
        
        #the target we are using to train the algo
        train_target=data[outcome].iloc[train]
        
        #training the algo using predictors and target
        model.fit(train_predictors,train_target)
        
        #record error from each corss-validation run
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
        
        print("Cross-validation score : %s" % "{0:.3%}".format(np.mean(error)))
        
        #fit the model again so that it can be refered outside the function
        model.fit(data[predictors],data[outcome])
    
    
    
#%%Logistic Regression

outcome_var='Loan_Status'
model=LogisticRegression()
predictor_var=['Credit_History']
classification_model(model,df,predictor_var,outcome_var)


#%%we can try different combination of variables
predictor_var=['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model,df,predictor_var,outcome_var)    
    
   
#%%Decision Tree

model=DecisionTreeClassifier()
predictor_var=['Credit_History','Gender','Married','Education']
classification_model(model,df,predictor_var,outcome_var)


#%%we can try different combination of variables

predictor_var=['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model,df,predictor_var,outcome_var)

#%%random forest

model=RandomForestClassifier(n_estimators=100)
predictor_var=['Gender','Married','Dependents','Education','Self_Employed',
               'Loan_Amount_Term','Credit_History','Property_Area','LoanAmount_log','TotalIncome_log']
classification_model(model,df,predictor_var,outcome_var)

#%%create a series with feature importances

featimp=pd.Series(model.feature_importances_,index=predictor_var).sort_values(ascending=False)
print(featimp)
    
#%%

model=RandomForestClassifier(n_estimators=25,min_samples_split=25,max_depth=7,max_features=1)
predictor_var=['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model,df,predictor_var,outcome_var)

#%%
    
    
    
    
    
    
    
    
    
    
    
    
    