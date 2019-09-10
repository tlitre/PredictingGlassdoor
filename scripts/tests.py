#!/usr/bin/env python
# coding: utf-8

# In[5]:


from time import time
import pandas as pd
import re
import numpy as np
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from ast import literal_eval


# In[6]:


#globals


# In[7]:


#read csv into pandas
df = pd.read_csv("glassdoorclean.csv")


# In[8]:


def initial_clean():
    del df['Listing ID']
    del df['Listing State']
    del df['HQ State']
    del df['Company']
    df.dropna()


# In[9]:


def get_seniority():
    seniorityArr = ['principal', 'manager', 'junior', 'jr', 'jr.', 'lead', 'snr', 'senior', 'sr', 'sr.', 'director', 'associate', 'i', 'ii', 'intern', 'iii', 'iv', '1', '2', '3', '4', 'grad']
    seniority = {
        'manager': ['manager'],
        'junior' : ['junior', 'jr'],
        'principal' : ['principal'],
        'director': ['director'],
        'lead': ['lead'],
        'senior' : ['senior', 'sr', 'snr'],
        'intern' : ['intern'],
        'associate': ['associate'],
        'one' : ['i', '1', ],
        'two' : ['ii', '2'],
        'three' : ['iii', '3'],
        'four' : ['iv', '4'],
        'grad' : ['grad', 'graduate'],
    }
    
    position = []
    for title in df['Job Title']:
        t = re.split(r'\W+', title)
        res = ""
        if not set(t).isdisjoint(seniorityArr):
            for name, ls in seniority.items():
                if not set(t).isdisjoint(ls):
                    res = name
                    break
        else:
            res = "none"
        position.append(res)
    df['Seniority'] = position
    del df['Job Title']


# In[10]:


def parse_description():
    desc = []
    for row in df['Description']:
        desc.append(literal_eval(row))
    df['Description'] = desc


# In[11]:


def get_average_salary():
    df['Avg Salary'] = 1000 * ((df['Salary Low (K)'] + df['Salary High (K)']) / 2)
    del df['Salary Low (K)']
    del df['Salary High (K)']


# In[12]:


##https://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column
def make_rating_ranges(df):
    conditions = [
        4.5 <= df['Company Rating'],
        (4 <= df['Company Rating']) & (df['Company Rating'] < 4.5),
        (3.5 <= df['Company Rating']) & (df['Company Rating'] < 4),
        (3 <= df['Company Rating']) & (df['Company Rating'] <= 3.5),
        df['Company Rating'] < 3
    ]
    choices = [5,4,3,2,1]
    df['Rating'] = np.select(conditions, choices, default='0')
    df.drop(df[df['Rating'] == '0'].index, inplace=True)


# In[13]:


initial_clean()
get_seniority()
parse_description()
get_average_salary()
make_rating_ranges(df)
df


# In[10]:


df.groupby('Rating').count()


# In[ ]:





# In[ ]:





# In[ ]:


def word_cloudify():
#make a word cloud to get a feel for what we're working with
#https://www.geeksforgeeks.org/generating-word-cloud-python/
#https://python.gotrained.com/text-classification-with-pandas-scikit/
    total_str = ""
    for listing in df['Description']:
        total_str += ' '.join(listing)
        wordcloud = WordCloud(width = 800, height = 800, 
                        background_color ='white',  
                        min_font_size = 12).generate(total_str)

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()
    
word_cloudify()


# In[14]:


def bag_words(data):
    bag = {}
    word_count = 0
    for row in data['Description']:
        for word in row:
            if word not in bag:
                bag[word] = word_count
                word_count += 1
    
    return bag
 
bag = bag_words(df)


# In[15]:


def get_features(description, bag):
    features = [0 for x in range(len(bag))]
    for word in description:
        ind = bag[word]
        features[ind] += 1
    return features
 


# In[16]:


def get_all_features(data, bag):
    all_features = []
    for description in data['Description']:
        all_features.append(get_features(description, bag))
    return all_features

all_features = get_all_features(df, bag)


# In[17]:


def export_feature_csv(df):
    export = df.filter(['Rating'], axis = 1)
    export['Features'] = all_features
    export.to_csv('withfeatures.csv', encoding='utf-8', index=False)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(all_features, df['Rating'], test_size=0.33, random_state=0)


# In[19]:


#saved for comparison
print('zeroR accuracy: .387')
print('Perceptron(max_iter=50) accuracy: 0.653')
print('MultinomialNB(alpha=.01) accuracy: 0.695')
print('MultinomialNB(alpha=.001) accuracy: 0.702')
print('BernoulliNB(alpha=.01) accuracy: 0.685')
print('BernoulliNB(alpha=.001) accuracy: 0.698')
print('RandomForestClassifier() accuracy: 0.586')
print('LogisticRegression() accuracy: 0.672')


# In[22]:


#clf = Perceptron(max_iter=50)
#clf = MultinomialNB(alpha=0.001)
clf = BernoulliNB(alpha=.0001)
#clf = RandomForestClassifier()
#clf = LogisticRegression()

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)


# In[21]:


print('Training size: {} Test Size: {}'.format(len(X_train), len(X_test)))


# In[ ]:





# In[ ]:




