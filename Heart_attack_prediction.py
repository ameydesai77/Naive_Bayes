#!/usr/bin/env python
# coding: utf-8
Framingham is a longitudinal cohort study, a type of epidemiological study
that follows a group of individuals over time to determine the natural history of certain diseases, explore the behavior
of those diseases, and identify the factors that might explain their development. #here since the objective is to check whether a person may suffer with heart related diseases or not ,
#is a classical example of classification ML model let us  use Naive Bayes classifier model to analyse the data
# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.style
plt.style.use('classic')


# ### Import Dataset

# In[2]:


data = pd.read_csv(r"C:\Users\AmEy\personal\ML_Series\DATASETS\framingham.csv",encoding='latin1')
data.head(3)


# ### Renaming column heads

# In[3]:


data.rename(columns={'TenYearCHD':'Heart_Attack'},inplace=True)


# In[4]:


data.columns


# ##### Checking shape of dataset

# In[5]:


print('No of Rows:',data.shape[0])
print('No of Columns:',data.shape[1])


# #### checking the datatype of the features

# In[6]:


data.info()


# In[7]:


data['education'].value_counts()


# In[8]:


# convert all datatype which are in float to type object 
cat = ['male','education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes','Heart_Attack']
for i in cat:
    data[i]=data[i].astype('object')


# In[9]:


data.info()


# In[10]:


data.head(3)


# #### Making different list for categorical columns and numerical columns
# 

# In[11]:


cat=[]
num=[]
for i in data.columns:
    if data[i].dtypes=='object':
        cat.append(i)
    else:
        num.append(i)
print('Categorical features are:    ',   cat)
print('Numerical features are : ',num)


# #### Describe for the numerical and categorical columns

# In[12]:


data[num].describe().T


# In[13]:


data[cat].describe()


# #### Unique values for Categorical variables

# In[14]:


for i in cat:
    print(i.upper())
    print(data[i].value_counts(),'\n')


# #### check for duplicate records

# In[15]:


dups = data.duplicated()
print('Number of duplicate rows = %d' %(dups.sum()))
#data[dups]


# In[16]:


data.isnull().sum()


# In[17]:


data['male'].value_counts()


# #### imputing missing values
data.Gender.mode()
data.Gender = data.Gender.fillna('female')

# In[18]:


plt.figure(figsize=(10,5))
sns.boxplot(data['age'],orient="v")
plt.show()


# In[19]:


# no possible outliers are present here,we will use mean to impute the missing value


# In[20]:


data.age.mean()


# In[21]:


# data.age.fillna(49.58)


# In[22]:


for column in ['education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes']:
    data[column].fillna(data[column].mode()[0],inplace=True)


# In[23]:


data.info()


# we will be converting these features again in categorical form as there are chances of changing there datatype

# In[24]:


cat1=['education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes']


# In[25]:


for i in cat1:
    data[i] = data[i].astype('object')


# In[26]:


plt.figure(figsize=(10,5))
sns.boxplot(data.cigsPerDay)
plt.show()


# In[27]:


# some possible outliers and hence median would be used for imputation
data['cigsPerDay'].median()


# In[28]:


data.cigsPerDay = data.cigsPerDay.fillna(0.0)


# In[29]:


plt.figure(figsize=(10,5))
sns.boxplot(data.totChol)
plt.show()


# In[30]:


# too many outliers ,hence median is used for imputation
data['totChol'].median()


# In[31]:


data.totChol=data.totChol.fillna(234.0)


# In[32]:


plt.figure(figsize=(10,5))
sns.boxplot(data.sysBP)
plt.show()


# In[33]:


# too many outliers ,hence median is used for imputation
data['sysBP'].median()


# In[34]:


data.sysBP=data.sysBP.fillna(128)


# In[35]:


plt.figure(figsize=(10,5))
sns.boxplot(data.diaBP)
plt.show()


# In[36]:


data['diaBP'].median()


# In[37]:


data.diaBP=data.diaBP.fillna(82)


# In[38]:


plt.figure(figsize=(10,5))
sns.boxplot(data.BMI)
plt.show()


# In[39]:


data['BMI'].median()


# In[40]:


data.BMI=data.BMI.fillna(25.41)


# In[41]:


plt.figure(figsize=(10,5))
sns.boxplot(data.glucose)
plt.show()


# In[42]:


data['glucose'].median()


# In[43]:


data.glucose=data.glucose.fillna(80)


# In[44]:


plt.figure(figsize=(10,5))
sns.boxplot(data.heartRate)
plt.show()


# In[45]:


data['heartRate'].median()


# In[46]:


data.heartRate=data.heartRate.fillna(75)


# In[47]:


data.isnull().sum()


# In[48]:


# all the null values are appropriately imputed

#recheck the unique values
for column in data.columns:
    if data[column].dtype =='object':
        print(column.upper(),': ',data[column].nunique())
        print(data[column].value_counts().sort_values())
        print('\n')
# ## Univariate Analysis

# In[49]:


fig,axes = plt.subplots(nrows=8,ncols=2)
fig.set_size_inches(15,20)
# Age
a = sns.distplot(data['age'],ax=axes[0][0])
a.set_title('Age Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['age'],orient = 'v',ax = axes[0][1])
a.set_title('Age Distribution_Boxplot',fontsize = 10)
# Cig_per_Day
a = sns.distplot(data['cigsPerDay'],ax=axes[1][0])  #ax=axes[i][0]
a.set_title('cigsPerDay Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['age'],orient = 'v',ax = axes[1][1]) #ax=axes[i][j]
a.set_title('cigsPerDay Distribution_Boxplot',fontsize = 10)
#'tot_Chol'
a = sns.distplot(data['totChol'],ax=axes[2][0])
a.set_title('totChol Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['totChol'],orient = 'v',ax = axes[2][1])
a.set_title('totChol Distribution_Boxplot',fontsize = 10)
# sysBP
a = sns.distplot(data['sysBP'],ax=axes[3][0])  #ax=axes[i][0]
a.set_title('sysBP Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['sysBP'],orient = 'v',ax = axes[3][1]) #ax=axes[i][j]
a.set_title('sysBP Distribution_Boxplot',fontsize = 10)
# diaBP
a = sns.distplot(data['diaBP'],ax=axes[4][0])
a.set_title('diaBP Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['diaBP'],orient = 'v',ax = axes[4][1])
a.set_title('diaBP Distribution_Boxplot',fontsize = 10)
# BMI
a = sns.distplot(data['BMI'],ax=axes[5][0])  #ax=axes[i][0]
a.set_title('BMI Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['BMI'],orient = 'v',ax = axes[5][1]) #ax=axes[i][j]
a.set_title('BMI Distribution_Boxplot',fontsize = 10)
#''heartRate''
a = sns.distplot(data['heartRate'],ax=axes[6][0])
a.set_title('heartRate Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['heartRate'],orient = 'v',ax = axes[6][1])
a.set_title('heartRate Distribution_Boxplot',fontsize = 10)
# 'glucose'
a = sns.distplot(data['glucose'],ax=axes[7][0])  #ax=axes[i][0]
a.set_title('glucose Distribution_Distplot',fontsize = 10)

a=sns.boxplot(data['glucose'],orient = 'v',ax = axes[7][1]) #ax=axes[i][j]
a.set_title('glucose Distribution_Boxplot',fontsize = 10)


# ## Bivariate and multivariate Analysis

# In[50]:


print(num)


# In[51]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['age'],jitter=True)
plt.show()

# younger people have less probability of heart Attack ,this pattern is clearly visible
# however probability of heart attack is low even for old age people,as per the above stiplet
# In[52]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['cigsPerDay'],jitter=True)
plt.show()


# In[53]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['totChol'],jitter=True)
plt.show()


# In[54]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['sysBP'],jitter=True)
plt.show()


# In[55]:


# for low systolic BP less chances of heart ,however even if sysBP is high,chances of heart attack is low


# In[56]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['diaBP'],jitter=True)
plt.show()


# In[57]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['BMI'],jitter=True)
plt.show()


# In[58]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['heartRate'],jitter=True)
plt.show()


# In[59]:


# lesser heart rate ,less chances of heart attack ,very clear pattern visible


# In[60]:


plt.figure(figsize=(8,5))
sns.stripplot(data['Heart_Attack'],data['glucose'],jitter=True)
plt.show()


# In[61]:


# Higher value of glucose higher chances of heart attack


# # same is plotted for all num features

# ### Correlation plot

# In[62]:


data.corr()


# In[63]:


sns.heatmap(data.corr(),annot=True)

# SysBP and diasBP are highly correlated ,so one of them would be dropped
# In[64]:


data.drop(['sysBP'],axis=1,inplace=True)


# In[65]:


sns.heatmap(data.corr(),annot=True)


# In[66]:


#sns.pairplot(data)


# In[67]:


print(cat)


# In[68]:


data = pd.get_dummies(data,columns=cat,drop_first=True)


# In[69]:


data.head()


# ### check Outliers

# In[70]:


plt.figure(figsize=(10,10))
data.iloc[:,:7].boxplot(vert=0)
plt.show()


# ### Outlier Treatment

# In[71]:


def remove_outliers(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_Range = Q1-(1.5 * IQR)
    upper_range = Q3+(1.5 * IQR)
    return lower_Range,upper_range


# In[72]:


for column in data.iloc[:,:7].columns:
    lr,ur = remove_outliers(data[column])
    data[column]=np.where(data[column]>ur,ur,data[column])
    data[column]=np.where(data[column]<lr,lr,data[column])


# In[73]:


plt.figure(figsize=(10,10))
data.iloc[:,:7].boxplot(vert=0)

# exclude correlated variable- sysBP 
# In[74]:


print(num)


# In[75]:


num1 = ['age','cigsPerDay','totChol','diaBP','BMI','heartRate','glucose']


# ### Scaling the variables with min-max technique(NORMALIZATION TECHNIQUE)

# In[76]:


data[num1] = data[num1].apply(lambda x:(x-x.min())/(x.max()-x.min()))


# In[77]:


data.head(3)


# In[78]:


# TRAIN-TEST SPLIT


# In[79]:


X = data.drop('Heart_Attack_1',axis = 1)
y = data['Heart_Attack_1']


# In[80]:


X.head(3)


# In[81]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state=664)


# ### Naive Bayes Model

# In[82]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# In[83]:


NB_Model = GaussianNB()

NB_Model.fit(X_train,y_train)


# In[84]:


# performance matrix on train dataset
y_train_pred = NB_Model.predict(X_train)
model_score = NB_Model.score(X_train,y_train)
print('Train score is:',model_score*100)
print(metrics.confusion_matrix(y_train,y_train_pred))

print(metrics.classification_report(y_train,y_train_pred))


# In[85]:


# performance matrix on train dataset
y_test_pred = NB_Model.predict(X_test)
model_score = NB_Model.score(X_test,y_test)
print('Test score is:',model_score*100,'\n')
print('Confusion Matrix:\n\n',metrics.confusion_matrix(y_test,y_test_pred))

print('classification Report:\n\n',metrics.classification_report(y_test,y_test_pred))


# now let us check accuracy using Random_Forest 

# ### Random Forest

# In[86]:


from sklearn.ensemble import RandomForestClassifier

Rfc_model = RandomForestClassifier(n_estimators=30)

Rfc_model.fit(X_train,y_train)


# In[87]:


y_pred1 = Rfc_model.predict(X_test)


# In[88]:


from sklearn import metrics
metrics.accuracy_score(y_test,y_pred1)


# In[89]:


cm1 = metrics.confusion_matrix(y_test,y_pred1)
cm1


# In[ ]:





# In[ ]:




