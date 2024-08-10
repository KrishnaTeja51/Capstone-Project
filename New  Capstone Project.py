#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score,recall_score,confusion_matrix


# In[7]:


#2. Load Data
data=pd.read_csv('creditcard (1).csv')
print(data.head())


# In[12]:


#3. Pre Process Data
# Separate features and target variable
x= data.drop('Class',axis=1) 
y=data['Class'] 
# Standardize the 'normal' feature
scaler=StandardScaler()
x['amount'] = scaler.fit_transform(x[['Amount']])


# In[13]:


#4.Split Data into Training and Testing Sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[14]:


#5.Train the Naive Bayes model
model=GaussianNB()
#Train the model
model.fit(x_train,y_train)


# In[15]:


#6. Make Predictions
#Make predictions on the test set
y_pred=model.predict(x_test)


# In[16]:


#7.Evaluate Model Performance
#Calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy:.2f}')
#Calculate precision
precision=precision_score(y_test,y_pred)
print(f'Precision: {precision:.2f}')
#Calculate recall (sensitivity)
recall=recall_score(y_test,y_pred)
print(f'Recall: {recall:.2f}')
#Confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# In[17]:


#8.Cross-Validation
cv_scores=cross_val_score(model,x,y,cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {cv_scores.mean():.2f}')


# In[ ]:




