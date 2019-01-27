#!/usr/bin/env python
# coding: utf-8

# In[67]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve,auc,confusion_matrix,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[160]:


#getting the working directory
os.getcwd()


# In[161]:


#setting the working directory
os.chdir(r'C:\Users\user\Desktop\Python code')


# In[162]:


#Reading train and test data
train=pd.read_csv(r'C:\Users\user\Desktop\Project\Churn or not\Train_data.csv')
test=pd.read_csv(r'C:\Users\user\Desktop\Project\Churn or not\Test_data.csv')


# In[163]:


#Checking the first 10 obesrvation
train.head(10)


# In[164]:


train.shape
test.shape


# In[165]:


train.isnull().sum().sum() #there is no missing values


# In[166]:


train.info()


# In[167]:


train.describe() 


# In[168]:


#there are space in the column names so we have to remove the space
col_names=train.columns
train.columns=col_names.str.replace(" ","_")
test.columns=test.columns.str.replace(" ",'_')
train.columns


# In[169]:


#As phone number is not useful so we have to drop the phone number
train=train.drop(columns={'phone_number'})
test=test.drop(columns={'phone_number'})


# In[170]:


train.dtypes
test.dtypes
#we have to convert the area codes to factor variable
train['area_code']=train['area_code'].astype('object')
test['area_code']=test['area_code'].astype('object')


# In[171]:


plt.figsize=(2,2)
sns.countplot(train['Churn'])
plt.xlabel('Customer Churn Type')
plt.ylabel('Frequency')
plt.title('Customer Churn Count')
#Our target variable suffers from class imbalance problem


# In[172]:


sns.countplot(x='area_code',hue='Churn',data=train)


# In[173]:


train.groupby(['state','Churn']).size().unstack(level=-1).plot(kind='bar', figsize=(20,6))


# In[174]:


sns.countplot(x='voice_mail_plan',hue='Churn',data=train)


# In[175]:


sns.countplot(x='international_plan',hue='Churn',data=train)


# In[176]:


train.plot(x='total_day_minutes',y='total_day_calls',kind='scatter')


# In[177]:


train.plot(x='total_day_charge',y='total_day_minutes',kind='scatter')
#Highly +vely corealted


# In[178]:


train.plot(x='total_night_charge',y='total_night_minutes',kind='scatter')
#Highly +vely corealted


# In[179]:


train.plot(x='total_eve_charge',y='total_eve_minutes',kind='scatter')
#Highly +vely corelated


# In[180]:


train.hist(figsize=(20,20))
plt.show()


# In[181]:


numeric_var=train.select_dtypes(exclude='object').columns
factor_var=train.select_dtypes(include='object').columns


# In[182]:


#Assigning levels to factor variable in the train dataset
for i in factor_var:
    train.loc[:,i]=pd.Categorical(train.loc[:,i])
    train.loc[:,i]=train.loc[:,i].cat.codes
    train.loc[:,i]=train.loc[:,i].astype('object')


# In[183]:


#Assigning label to the factor variable in the test dataset
for i in factor_var:
    test.loc[:,i]=pd.Categorical(test.loc[:,i])
    test.loc[:,i]=test.loc[:,i].cat.codes
    test.loc[:,i]=test.loc[:,i].astype('object')


# In[184]:


sns.boxplot(train['account_length'])


# In[185]:


sns.boxplot(train['number_vmail_messages'])


# In[186]:


sns.boxplot(train['total_day_minutes'])


# In[187]:


sns.boxplot(train['total_day_calls'])


# In[188]:


sns.boxplot(train['total_eve_calls'])


# In[189]:


sns.boxplot(train['total_night_calls'])


# In[98]:


# #Finding outlier
#for random forest no outlier analysis required so i have commented the code
# for i in numeric_var:
#     q75,q25=np.percentile(train[i],[75,25])
#     iqr=q75-q25
#     upper=q75+(1.5*iqr)
#     lower=q25-(1.5*iqr)
#     train.loc[(train[i] < lower) | (train[i] > upper),i]=np.nan


# In[145]:


train.isnull().sum().sum()


# In[100]:


for i in numeric_var:
    train.loc[:,i]=train.loc[:,i].fillna(train[i].median())


# In[152]:


#multicolinarity
x=train[numeric_var].corr()
plt.figure(figsize=(15,10))
sns.heatmap(x,annot=True,cmap='viridis',linewidths=4)


# In[153]:


upper=x.where(np.triu(np.ones(x.shape),k=1).astype(np.bool))
to_drop=[column for column in upper.columns if any (upper[column] > 0.70)]


# In[154]:


for i in factor_var:
    chi2,p,dof,ex=chi2_contingency(pd.crosstab(train['Churn'],train[i]))
    print(i)
    print(p)


# In[155]:


drop_var=['area_code']
to_drop=to_drop+drop_var


# In[156]:


train=train.drop(to_drop,axis=1)
test=test.drop(to_drop,axis=1)


# In[106]:


#Feature scaling
numeric_var=train.select_dtypes(exclude='object').columns
for i in numeric_var:
    train[i]=(train[i]-(train[i].mean()))/train[i].std()


# In[107]:


for i in numeric_var:
    test[i]=(train[i]-(train[i].mean()))/train[i].std()


# In[108]:


#To check accuracy,recall..etc
def result (x_test,y_test,model):
    predict=model.predict(x_test)
    x=confusion_matrix(y_test.tolist(),predict)
    TN=x[0,0]
    TP=x[1,1]
    FP=x[0,1]
    FN=x[1,0]
    print(x)
    print('Accuracy = ',((TN+TP)/(TN+TP+FP+FN)))
    print('Recall = ',(TP/(TP+FN)))
    print('Specificity = ',(TN/(TN+FP)))
    print('False positive rate = ', (FP/(FP+TN)))
    print('False negative rate= ', (FN/(FN+TP)))
    print('auc score = ',roc_auc_score(y_test.tolist(),predict))


# In[109]:


def plot_roc_curve(fpr, tpr, label=None):
 plt.plot(fpr, tpr, linewidth=2, label=label)
 plt.plot([0, 1], [0, 1], 'k--')
 plt.axis([0, 1, 0, 1])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 


# In[110]:


train['Churn'].value_counts()


# In[190]:


#Over sampling using smote
x_train=train.drop('Churn',axis=1)
y_train=train['Churn']
Smo = SMOTE(random_state=101)
x_train_new, y_train_new = Smo.fit_sample(x_train,y_train)


# In[112]:


pd.Series(y_train_new).value_counts()


# In[113]:


###logistic regression
glm=LogisticRegression().fit(x_train_new,y_train_new.astype('int'))


# In[114]:


result(test.drop('Churn',axis=1),test['Churn'],glm)


# In[115]:


predict=glm.predict(test.drop('Churn',axis=1))
fpr, tpr, thresholds = roc_curve(test['Churn'].tolist(), predict)
plot_roc_curve(fpr,tpr)
plt.title('Logistic regression roc plot')


# In[116]:


###decission tree
dt=DecisionTreeClassifier().fit(x_train_new,y_train_new.astype('int'))


# In[117]:


result(test.drop('Churn',axis=1),test['Churn'],dt)


# In[118]:


predict=dt.predict(test.drop('Churn',axis=1))
fpr, tpr, thresholds = roc_curve(test['Churn'].tolist(), predict)
plot_roc_curve(fpr,tpr)
plt.title('Decission tree roc plot')


# In[119]:


#knn method
kn=KNeighborsClassifier(n_neighbors=5)
knn=kn.fit(x_train_new,y_train_new.astype('int'))


# In[120]:


result(test.drop('Churn',axis=1),test['Churn'],knn)


# In[121]:


predict=knn.predict(test.drop('Churn',axis=1))
fpr, tpr, thresholds = roc_curve(test['Churn'].tolist(), predict)
plot_roc_curve(fpr,tpr)
plt.title('Knn roc plot')


# In[122]:


sv_cl=svm.SVC(kernel='linear')
sv_cl.fit(x_train_new,y_train_new.astype('int'))


# In[123]:


result(test.drop('Churn',axis=1),test['Churn'],sv_cl)


# In[124]:


svm_rad=svm.SVC(kernel='rbf',gamma='auto')
svm_rad.fit(x_train_new,y_train_new.astype('int'))


# In[125]:


result(test.drop('Churn',axis=1),test['Churn'],svm_rad)


# In[126]:


#Ranodo forest
rf=RandomForestClassifier().fit(x_train_new,y_train_new.astype('int'))


# In[127]:


result(test.drop('Churn',axis=1),test['Churn'],rf)


# In[128]:


predict=rf.predict(test.drop('Churn',axis=1))
fpr, tpr, thresholds = roc_curve(test['Churn'].tolist(), predict)
plot_roc_curve(fpr,tpr)
plt.title('random forest roc plot')


# In[129]:


#so among all the above model random forest is the best model. So lets tune random forest
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[130]:


#the code is taking so much time
# rf=RandomForestClassifier()
# rf1 = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,
#                                cv = 3, verbose=2, random_state=42)
# rf1.fit(x_train_new,y_train_new.astype('int'))


# In[191]:


# Training Final Model With Optimum Parameters
final_model = RandomForestClassifier(random_state=101, n_estimators = 400,n_jobs=-1)
final_model.fit(x_train_new,y_train_new.astype('int'))


# In[192]:


result(test.drop('Churn',axis=1),test['Churn'],final_model)


# In[193]:


predict=final_model.predict(test.drop('Churn',axis=1))
fpr, tpr, thresholds = roc_curve(test['Churn'].tolist(), predict)
plot_roc_curve(fpr,tpr)
plt.title('random forest roc plot')


# In[ ]:


#So our model accuracy is 96.
#Specificity is 98
#Recall is 78.
#auc score is 88.
#so random forest is the best model for the given dataset

