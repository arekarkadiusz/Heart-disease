#!/usr/bin/env python
# coding: utf-8

# ## About dataset:
# 
# WHO has estimated 12 million deaths occur worldwide, every year due to Heart diseases. Half the deaths in the United States and other developed countries are due to cardio vascular diseases. The early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high risk patients and in turn reduce the complications. This research intends to pinpoint the most relevant/risk factors of heart disease as well as predict the overall risk using logistic regression.

# In[1]:


import pandas as pd
import numpy as np


# ### Import and checking dataset

# In[2]:


raw_data = pd.read_csv('C:/Users/arkad/Desktop/Pliki_do_analizy/heat_disease_ds_kaggle/framingham.csv')
raw_data.head()


# In[3]:


raw_data.info()


# In[4]:


raw_data.describe().T


# ## Checking correlation with target column 'TenYearCHD'

# In[5]:


raw_data.corrwith(raw_data.TenYearCHD)*100


# In[6]:


(raw_data.corrwith(raw_data.TenYearCHD)*100).plot(kind='bar')


# # Plotting charts in order to see variable distribution

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


raw_data.columns


# In[9]:


cols =['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
plt.figure(figsize=(20,20))
for i in range(1,17):
    plt.subplot(4,4,i)
    sns.histplot(raw_data[cols[i-1]])


# # Comparing the data in each column with and without cases of heart disease

# In[10]:


cols = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']
plt.figure(figsize=(20,20))
for i in range (1,17):
    plt.subplot(4,4,i)
    sns.distplot(raw_data[raw_data['TenYearCHD']==1][cols[i-1]],hist=False,color='red')
    sns.distplot(raw_data[raw_data['TenYearCHD']==0][cols[i-1]],hist=False,color='blue')


# # Checking average and median value with and without heart disease

# In[11]:


raw_data.mean()


# In[12]:


raw_data.median()


# In[13]:


raw_data.groupby(by='TenYearCHD').mean()


# In[14]:


raw_data.groupby(by='TenYearCHD').median()


# In[15]:


raw_data['TenYearCHD'].value_counts()


# ### Checking missing values and fill or remove empty rows

# In[16]:


raw_data.isnull().sum()[raw_data.isnull().sum()>0]


# In[17]:


percentage_empty_rows=raw_data.isnull().sum()[raw_data.isnull().sum()>0]


# In[18]:


percentage_empty_rows=(percentage_empty_rows/4238)*100


# In[19]:


percentage_empty_rows


# ### There are not many empty rows releating to total dataset so filling empty rows by mean or median shouldn't mess up data distribution. Make copy of dataset

# In[20]:


data = raw_data.copy()


# In[21]:


data.head()


# In[22]:


data['education_mean'] = data['education'].fillna(data['education'].mean())
data['education_median'] = data['education'].fillna(data['education'].median())


# In[23]:


fig=plt.figure()
ax=fig.add_subplot()
data['education_mean'].plot.density(ax=ax,color='green')
data['education_median'].plot.density(ax=ax,color='blue')
data['education'].plot.density(ax=ax,color='red')


# In[24]:


data['cigsPerDay_mean'] = data['cigsPerDay'].fillna(data['cigsPerDay'].mean())
data['cigsPerDay_median'] = data['cigsPerDay'].fillna(data['cigsPerDay'].median())


# In[25]:


fig=plt.figure()
ax=fig.add_subplot(111)
data['cigsPerDay_mean'].plot.density(ax=ax,color='orange')
data['cigsPerDay_median'].plot.density(ax=ax,color='blue')
data['cigsPerDay'].plot.density(ax=ax,color='red')


# In[26]:


data['BPMeds_mean'] = data['BPMeds'].fillna(data['BPMeds'].mean())
data['BPMeds_median'] = data['BPMeds'].fillna(data['BPMeds'].median())


# In[27]:


fig=plt.figure()
ax=fig.add_subplot(111)
data['BPMeds_mean'].plot.density(ax=ax,color='green')
data['BPMeds_median'].plot.density(ax=ax,color='black')
data['BPMeds'].plot.density(ax=ax,color='red')


# In[28]:


data['totChol_mean']=data['totChol'].fillna(data['totChol'].mean())
data['totChol_median']=data['totChol'].fillna(data['totChol'].median())


# In[29]:


fig=plt.figure()
ax=fig.add_subplot(111)
data['totChol_mean'].plot.density(ax=ax,color='black')
data['totChol_median'].plot.density(ax=ax,color='blue')
data['totChol'].plot.density(ax=ax,color='red')


# In[30]:


data['BMI_mean']=data['BMI'].fillna(data['BMI'].mean())
data['BMI_median']=data['BMI'].fillna(data['BMI'].median())


# In[31]:


fig=plt.figure()
ax=fig.add_subplot(111)
data['BMI_mean'].plot.density(ax=ax,color='black')
data['BMI_median'].plot.density(ax=ax,color='blue')
data['BMI'].plot.density(ax=ax,color='red')


# In[32]:


data['glucose_mean']=data['glucose'].fillna(data['glucose'].mean())
data['glucose_median']=data['glucose'].fillna(data['glucose'].median())


# In[33]:


fig=plt.figure()
ax=fig.add_subplot(111)
data['glucose_mean'].plot.density(ax=ax,color='black')
data['glucose_median'].plot.density(ax=ax,color='blue')
data['glucose'].plot.density(ax=ax,color='red')


# ### For heartRate column empty row will be filled by median due to only one empty row.

# In[34]:


data['heartRate_median'] = data['heartRate'].fillna(data['heartRate'].median())


# ##### After checking at charts there is not data distribution changes. Charts show that median is covered with data well. To compare previous data with data filled by mode, correlation is used.

# In[35]:


plt.figure(figsize=(15,15))
sns.heatmap(data[['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose',
      'education_median','cigsPerDay_median','BPMeds_median','totChol_median','BMI_median','heartRate','glucose_median']].corr(),annot=True)


# ##### Correlation with no changes before and now

# ## Creating new data frame with filled empty rows

# In[36]:


data.columns


# In[37]:


filled_data = data[['male', 'age', 'education_median', 'currentSmoker', 'cigsPerDay_median', 
       'prevalentHyp', 'totChol_median', 'sysBP','BPMeds_median','prevalentStroke','diabetes',
       'diaBP', 'BMI_median', 'heartRate_median', 'glucose_median', 'TenYearCHD']]


# In[38]:


filled_data.head()


# In[39]:


filled_data.info() # checking if each empty row is filled


# ### Each column shows corrected value. Now outliers are checked

# In[40]:


plt.figure(figsize=(15,15))
filled_data.boxplot()


# In[41]:


filled_data.columns


# ### Removing outliers using z-score

# In[42]:


from scipy import stats

z_score_data = np.abs(stats.zscore(filled_data))


# In[43]:


z_score_data


# In[44]:


threshold = 4
np.where(z_score_data > threshold)


# In[45]:


filled_data_z_score = filled_data[(z_score_data < threshold).all(axis = 1)]


# In[46]:


filled_data_z_score.head()


# In[47]:


plt.figure(figsize=(15,15))
filled_data_z_score.boxplot()


# ### After removing outliers now variable distribution is checking

# In[48]:


filled_data_z_score.columns


# In[49]:


cols =['male', 'age', 'education_median', 'currentSmoker', 'cigsPerDay_median',
       'prevalentHyp', 'totChol_median', 'sysBP', 'BPMeds_median',
       'prevalentStroke', 'diabetes', 'diaBP', 'BMI_median',
       'heartRate_median', 'glucose_median', 'TenYearCHD']

plt.figure(figsize=(20,20))
for i in range(1,17):
    plt.subplot(4,4,i)
    sns.distplot(filled_data_z_score[cols[i - 1]], hist = False)


# #### Data can be split at test and train dataset

# In[50]:


from sklearn.model_selection import train_test_split


# In[51]:


X = filled_data_z_score
y = filled_data_z_score.pop('TenYearCHD')


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[53]:


print(f'Train set size: {X_train.shape}')
print(f'Test set size: {X_test.shape}')
print(f'Target train set size: {y_train.shape}')
print(f'Target test size: {y_test.shape}')


# #### In order to increase accuracy of  model, StandardScaler will be used

# In[54]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Logistic regression

# In[55]:


from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(max_iter=3000)
logReg.fit(X_train, y_train)


# #### Predicition based on model

# In[56]:


y_pred = logReg.predict(X_test)
y_pred[:30]


# #### Model evaluation

# In[57]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# ##### Confusion Matrix

# In[58]:


confusion_matrix(y_test, y_pred)


# In[79]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ## SVC

# In[60]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[61]:


classifier = SVC(kernel='linear')
param_grid = {'C':np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])}
grid_search = GridSearchCV(classifier,param_grid,scoring='accuracy',cv=5)
grid_search.fit(X_train, y_train)


# In[62]:


grid_search.best_params_


# In[63]:


svc = SVC(kernel='linear', C=1)
svc.fit(X_train,y_train)


# In[64]:


svc_pred = svc.predict(X_test)


# In[65]:


svc_pred[:30]


# In[66]:


confusion_matrix(y_test,svc_pred)


# In[67]:


print(f'Accuracy: {accuracy_score(y_test, svc_pred)}')


# In[68]:


confusion_matrix(y_test, y_pred)


# In[69]:


print(classification_report(y_test,svc_pred, zero_division=1))


# ## Decision Tree

# In[70]:


from sklearn.tree import DecisionTreeClassifier


# In[71]:


tree_classifier = DecisionTreeClassifier()
param_grid_tree = {'max_depth':[1,2,3,4,5,6,7,8,9],
                  'criterion':['gini','entropy'],
                  'min_samples_leaf':[2,3,4,5,6,7,8,9,10]}
grid_search_tree = GridSearchCV(tree_classifier,param_grid_tree,scoring='accuracy',cv=5)
grid_search_tree.fit(X_train, y_train)


# In[72]:


grid_search_tree.best_params_


# In[73]:


tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=6)
tree_model.fit(X_train, y_train)


# In[74]:


y_pred_tree=tree_model.predict(X_test)
y_pred_tree[:30]


# In[75]:


confusion_matrix(y_test, y_pred_tree)


# In[76]:


print(f'Accuracy: {accuracy_score(y_test, y_pred_tree)}')


# In[78]:


print(classification_report(y_test,y_pred_tree, zero_division=1))


# In[ ]:




