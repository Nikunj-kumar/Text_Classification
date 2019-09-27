#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
news_train = skd.load_files("Desktop/ML_Project/News_Dataset/20news-bydate-train", categories=categories, encoding="ISO-8859-1")
news_test = skd.load_files("Desktop/ML_Project/News_Dataset/20news-bydate-test", categories=categories, encoding="ISO-8859-1")


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
counts_vect = CountVectorizer()
X_train_tf = counts_vect.fit_transform(news_train.data)
X_train_tf.shape


# In[3]:


from sklearn.feature_extraction.text import TfidfTransformer
tfid_transformer = TfidfTransformer()
X_train_tfid = tfid_transformer.fit_transform(X_train_tf)
X_train_tfid.shape


# In[4]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfid, news_train.target)


# In[6]:


X_test_tf = counts_vect.transform(news_test.data)
X_test_tfid = tfid_transformer.transform(X_test_tf)
predict = clf.predict(X_test_tfid)


# In[12]:


from sklearn import metrics
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(news_test.target, predict)), 
print(metrics.classification_report(news_test.target, predict, target_names=news_test.target_names))
print(metrics.confusion_matrix(news_test.target, predict))


# In[ ]:




