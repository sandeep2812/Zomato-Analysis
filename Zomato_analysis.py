#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import sqlite3


# In[4]:


sqlite3.connect(r'/Users/sandeepkumar/Downloads/Resources/zomato_rawdata.sqlite')


# In[5]:


con = sqlite3.connect(r'/Users/sandeepkumar/Downloads/Resources/zomato_rawdata.sqlite')


# In[6]:


pd.read_sql_query("SELECT * FROM Users", con).head(2)


# In[7]:


df = pd.read_sql_query("SELECT * FROM Users", con)


# In[8]:


df


# In[9]:


df.shape


# In[10]:


df.size


# In[11]:


df.ndim


# In[12]:


df.dtypes


# In[13]:


df.columns


# In[ ]:





# In[14]:


# deal with missing values


# In[15]:


df.head(2)


# In[16]:


df.isnull().sum()


# In[17]:


len(df)


# In[18]:


df.isnull().sum()/len(df)*100


# In[19]:


df['rate'].unique()


# '''
#      -->> As we notice around 50 % of data will be lost if we delete the nan values in dish_liked column
#          We will keep that column for now..
#         
#      -->> let's check for rate column as it contains 15% of its points as null value which 
#          is one of the most important feature.. 
#          
# '''

# In[27]:


df['rate'].replace(('NEW', '-'), np.nan, inplace=True)


# In[29]:


df['rate'].unique()


# In[30]:


#we need 4.1 not 4.1/5


# In[31]:


"4.1/5".split('/')[0]


# In[32]:


type("4.1/5".split('/')[0])


# In[33]:


float("4.1/5".split('/')[0])


# In[37]:


df['rate'] = df['rate'].apply(lambda x : float(x.split('/')[0]) if type(x)==str else x)


# In[38]:


df['rate']


# # relation between online order option and rating of the restaurant ?
# 

# In[41]:


x=pd.crosstab(df['rate'], df['online_order'])


# In[42]:


x


# In[44]:


x.plot(kind='bar', stacked=True)


# In[45]:


x


# In[47]:


x.sum(axis=1).astype(float)


# In[48]:


## we need Floating division of dataframe or normalized values of x dataframe across rows..just call x.div() 
#& set axis=0
## div is a in-built function of pandas designed for dataframe data-structure..


# In[49]:


x.div(x.sum(axis=1).astype(float),axis=0)


# In[50]:


normalize_df = x.div(x.sum(axis=1).astype(float),axis=0)


# In[51]:


normalize_df


# In[52]:


(normalize_df*100).plot(kind='bar', stacked = True)


# In[53]:


''' Inference ::

For good rating ie > 4 , for most instances it seems that rest who accepts online order have received more number of ratings than those rest. who don't accept online order

'''

4.. Data Cleaning to perform Text Analysis
Perform Text Analysis.. ie analysing customer reviews of Quick Bites restaurant : 
ways to do it..
a) using wordcloud

    But wordcloud will not give a clear cut , how important words are
    So lets use a concept of frequency over here...
    
b) using plots/charts-- where each word have some frequency..
b) using plots/charts
We need Pre-processed data so that we can plot charts
ie  (Biryani , 10K)
    (Chicken , 15K)
    etc..
    
# In[54]:


df['rest_type'].unique()


# In[55]:


df['rest_type'].isnull().sum()


# In[57]:


data=df.dropna(subset=['rest_type'])


# In[58]:


data['rest_type'].isnull().sum()


# In[59]:


data['rest_type'].unique()


# In[60]:


### if we need whole data of 'Quick Bites' restaurant 


# In[61]:


### extracting data of 'Quick Bites' only ..


# In[62]:


data[data['rest_type'].str.contains('Quick Bites')]


# In[65]:


quick_bites_df = data[data['rest_type'].str.contains('Quick Bites')]


# In[66]:


quick_bites_df


# In[67]:


quick_bites_df.shape


# In[68]:


quick_bites_df.ndim


# In[69]:


quick_bites_df.dtypes


# In[70]:


quick_bites_df.columns


# In[72]:


quick_bites_df['menu_item'].unique()


# In[ ]:





# In[ ]:




 How to Perform Data Pre-processing to pre-process this data..
    Steps-->>
        a) Perform Lower-case operation
        b) Do tokenization
        c) Removal of stopwords from data
        d) Store your entire data in the list so that we can commpute frequency of each word
        e) Do plotting , using Unigram  , bigram & Trigram analysis..
# In[73]:


#a) Perform Lower-case operation


# In[74]:


quick_bites_df.head(2)


# In[76]:


quick_bites_df['reviews_list']


# In[78]:


quick_bites_df['reviews_list']=quick_bites_df['reviews_list'].apply(lambda x:x.lower())


# In[80]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[81]:


quick_bites_df['reviews_list']


# In[82]:


quick_bites_df['reviews_list'][3]


# In[ ]:





# In[83]:


#Do tokenization


# In[84]:


##  Creating a regular expression tokenizer that have only alphabets , ie remove all the special characters
# This will return separate words (tokens) from the text in the form of list


# In[91]:


from nltk.corpus import RegexpTokenizer


# In[92]:


tokenizer = RegexpTokenizer('[a-zA-Z]+')


# In[93]:


tokenizer


# In[94]:


## tokenize data of third review


# In[98]:


tokenizer.tokenize(quick_bites_df['reviews_list'][3])


# In[101]:


sample = data[0:10000]    #### u can consider some sample if u don't have good specifications in your system !


# In[104]:


reviews_tokens = sample['reviews_list'].apply(tokenizer.tokenize)


# In[105]:


reviews_tokens


# In[ ]:




Performing Unigram analysis & removal of stopwords ..
# In[106]:


#removal of stopwords ..


# In[107]:


reviews_tokens


# In[113]:


import nltk
nltk.download('stopwords')


# In[114]:


from nltk.corpus import stopwords


# In[117]:


stop = stopwords.words('english')


# In[118]:


print(stop)


# In[119]:


# Adding custom words to stopwords 


# In[120]:


stop.extend(['rated', 'n', 'nan', 'x', 'RATED', 'Rated'])


# In[121]:


print(stop)


# In[122]:


len(stop)


# In[ ]:




