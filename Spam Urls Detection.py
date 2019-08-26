
# coding: utf-8

# In[3]:


# EDA Packages
import pandas as pd
import numpy as np
import random


# In[4]:


# Machine Learning Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[5]:


# Load Url Data 
urls_data = pd.read_csv("C:/Users/leo/Downloads/Programs/data.csv")


# In[6]:


type(urls_data)


# In[7]:


urls_data.head()


# In[12]:


def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')  # make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')  # make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0, len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')  # make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))  # remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove(
            'com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens


# In[13]:


# Labels
y = urls_data["label"]


# In[14]:


# Features
url_list = urls_data["url"]


# In[15]:


# Using Default Tokenizer
# vectorizer = TfidfVectorizer()


# In[16]:


# Using Custom Tokenizer
vectorizer = TfidfVectorizer(tokenizer=makeTokens)


# In[17]:


# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Model Building
# using logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)


# In[20]:


# Accuracy of Our Model
print("Accuracy ", logit.score(X_test, y_test))


# In[40]:


X_predict = ["https://ocw.mit.edu/resources/res-3-004-visualizing-materials-science-fall-2017/student-projects-by-year/2015-MIT/image-processing-using-the-watershed-transformation/",
             "google.com/search=jcharistech",
             "google.com/search=faizanahmad",
             "pakistanifacebookforever.com/getpassword.php/",
             "www.radsport-voggel.de/wp-admin/includes/log.exe",
             "ahrenhei.without-transfer.ru/nethost.exe ",
             "www.itidea.it/centroesteticosothys/img/_notes/gum.exe",
            "https://internshala.com/internships/matching-preferences"]


# In[41]:


X_predict = vectorizer.transform(X_predict)
New_predict = logit.predict(X_predict)


# In[42]:


print(New_predict)


# In[24]:


# https://db.aa419.org/fakebankslist.php
X_predict1 = ["https://mail.google.com/mail/u/0/#inbox",
              "www.buyfakebillsonlinee.blogspot.com",
              "www.unitedairlineslogistics.com",
              "www.stonehousedelivery.com",
              "www.silkroadmeds-onlinepharmacy.com"]


# In[25]:


X_predict1 = vectorizer.transform(X_predict1)
New_predict1 = logit.predict(X_predict1)
print(New_predict1)


# In[26]:


# Using Default Tokenizer
vectorizer = TfidfVectorizer()


# In[27]:


# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


# Model Building
logit = LogisticRegression()  # using logistic regression
logit.fit(X_train, y_train)


# In[29]:


# Accuracy of Our Model with our Custom Token
print("Accuracy ", logit.score(X_test, y_test))

#  finished...!

