#!/usr/bin/env python
# coding: utf-8

# In[40]:


#data handling
import pandas as pd 
import re
import numpy as np  

#visualization
import matplotlib.pyplot as plt

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.feature_selection import mutual_info_classif
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter

#classification
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,recall_score,precision_score,f1_score 
from sklearn.metrics import roc_curve,auc
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

# performance metrics
from sklearn.metrics import balanced_accuracy_score,f1_score,precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score


# ## Data Reading, Cleaning and Merging

# In[43]:
def run_machine_learning_pipeline(data):
    # Assuming data is a DataFrame passed from app.py
    # Your existing machine learning pipeline code goes here

    # Example processing steps
    # data.shape or other operations can now be used here
    
    # Placeholder for pipeline result
    results = {"example_result": 42}  # Replace with actual results from your ML pipeline
    
    return results  # Ensure it returns something to display in Streamlit

print('rows,columns')
data.shape


# In[45]:


#print(targets.columns)
#print(topTable.columns)


# In[47]:


#check for missing values
datanul=data.isnull().sum()
g=[i for i in datanul if i>0]

print('columns with missing values:%d'%len(g))


# In[48]:


print("Do we have NA values?")
print(np.any(datadata.isna()))

print("In case we would have NA values, show these rows")
# Display the na values
display(data[np.any(data.isna(), axis=1)])
print("Empty rows if no NA values")

print("="*79)

print("If we would have NA values, drop the rows containing them")
data.dropna(inplace=True)


# In[49]:


#plot a bar chat to display the class distribution
data['HLAB27_status'].value_counts().plot.bar(figsize=(4,4))


# In[50]:


# Look at the correlations between multiple features by displaying a correlation plot (in heatmap form)
#f, ax = plt.subplots(figsize=(20, 8))
#corr= merged_data.corr()
#sns.heatmap(corr,
#            cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            mask=np.zeros_like(corr, dtype=bool),
#            square=True,
#            annot=True,
#            ax=ax)

