#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df = pd.read_csv("Dent.csv")


# In[8]:


# Check for missing values
print(df.isnull().sum())

# Handle missing values if necessary
df.fillna(df.mean(), inplace=True)  # Example: Replace missing values with mean


# In[4]:


# Check for missing values
print(df.isnull().sum())

# Replace missing values with mean for numeric columns only
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())


# In[9]:


X = df.drop(columns=['Gender', 'Sample ID', 'Sl No'])  # Independent variables
Y = df['Gender']  # Target variable


# In[10]:


# Example: Normalize the data
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
X_normalized = scaler.fit_transform(X)


# In[8]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# Initialize models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Assuming you have trained your model and predicted values
Y_pred = model.predict(X_test)  # Replace with your actual prediction

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable in training data
Y_train_encoded = label_encoder.fit_transform(Y_train)

# Transform the target variable in test data
Y_test_encoded = label_encoder.transform(Y_test)

# Encode predicted values
Y_pred_encoded = label_encoder.transform(Y_pred)
# Calculate accuracy, confusion matrix, and ROC AUC
accuracy = accuracy_score(Y_test_encoded, Y_pred_encoded)
cm = confusion_matrix(Y_test_encoded, Y_pred_encoded)
roc_auc = roc_auc_score(Y_test_encoded, Y_pred_encoded)


# In[11]:


pip install xgboost


# In[23]:


# Example: ROC curve and AUC for a model (e.g., Logistic Regression)
from sklearn.preprocessing import LabelEncoder

# Assuming Y_pred contains your predicted labels as strings ('Male', 'Female')
label_encoder = LabelEncoder()
Y_pred_encoded = label_encoder.fit_transform(Y_pred)



# In[24]:



roc_auc = roc_auc_score(Y_test_encoded, Y_pred_proba_positive)


# In[25]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Encode true labels (if not already encoded)
label_encoder = LabelEncoder()
Y_test_encoded = label_encoder.fit_transform(Y_test)

# Encode predicted labels
Y_pred_encoded = label_encoder.transform(Y_pred)

# Compute accuracy
accuracy = accuracy_score(Y_test_encoded, Y_pred_encoded)

# Compute confusion matrix
cm = confusion_matrix(Y_test_encoded, Y_pred_encoded)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(Y_test_encoded, Y_pred_proba_positive)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Print accuracy and confusion matrix
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)

# Document your project steps and results
# Write your project summary and findings in a file or report
# Include dataset description, preprocessing steps, model building process, evaluation metrics, conclusions, etc.
# Save the file for submission or future reference


# In[ ]:




