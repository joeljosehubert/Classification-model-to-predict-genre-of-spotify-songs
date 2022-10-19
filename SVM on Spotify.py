#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the necessary libraries (numpy and pandas)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Import the training dataset
train = pd.read_csv("/Users/noeljoseph/Downloads/Assignment_spotify_classification_problem_2022/CS98XClassificationTrain.csv")


# In[3]:


#dropping null values if any were present and checking the data
train = train.dropna()
train.head()


# In[4]:


# let's drop some object classes
train = train.drop(['Id', 'title', 'artist'], axis = 'columns') #dropping the varibale artists too eventhough a chi-square test revealed significant relationship between artists and genres
train.head()


# In[5]:


#checking the number of columns and rows
train.shape


# In[6]:


#dropping the label 'top genre'
x = train.drop(['top genre'], axis = 'columns')
x.head()


# In[7]:


#visualising for better undertanding of the spread of the variables
x.hist(bins=10, figsize=(20,15))
plt.show()


# In[8]:


y = train['top genre']
y.head()


# In[9]:


x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(x, y, test_size=0.25, random_state=42)


# In[10]:


x.head()


# In[11]:


x_train_set.shape


# In[12]:


x_train_set


# In[13]:


y_train_set.shape


# In[14]:


y_train_set


# In[15]:


# ovo_svc_clf = OneVsOneClassifier(SVC(kernel="rbf", gamma=5, C=1)) # e.g. does badly
ovo_svc_clf = OneVsOneClassifier(SVC(kernel="poly", degree=3, coef0=1, C=22)) # e.g. does better
ovo_svc_clf.fit(x_train_set, y_train_set)
preds = ovo_svc_clf.predict(x_test_set)
conf_mx = confusion_matrix(y_test_set, preds)
conf_mx


# In[16]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[17]:


test = pd.read_csv("/Users/noeljoseph/Downloads/Assignment_spotify_classification_problem_2022/CS98XClassificationTest.csv")


# In[18]:


test = test.dropna()
test.head()


# In[19]:


test = test.drop(['Id', 'title', 'artist'], axis = 'columns')
test.head()


# In[20]:


test.shape


# In[21]:


prediction = ovo_svc_clf.predict(test)


# In[22]:


print(prediction)


# In[23]:


prediction.flatten()


# In[24]:


results = pd.DataFrame(prediction)
print(results)


# In[25]:


pd.DataFrame(prediction).to_csv("sample11.csv")

