
# coding: utf-8

# # Topic Extraction

# In[17]:


import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
import pandas as pd
logging.basicConfig(level=logging.DEBUG)


# In[2]:


corpus_files = sorted(os.listdir('./corpus_per_president/'))


# # Defining main functions
# This data loader class will allow us to manage our data easily and focus on specific timelines or years. This is important since blindely looking at all data will incure in mixed topics that are related to all presidents throughout history. The usage of KMeans clustering along with term frequency–inverse document frequency feature extracture allows us to only focus on the important words and ignore the "useless" words such as "the" and "a" which do not provide essential information.

# In[115]:


class CorpusDataLoader(object):    
    def __init__(self):
        self.data_folder = '/home/native/projects/NLP/finalProj/corpus_per_president/'
        self.years=sorted(os.listdir(self.data_folder))
        self.km = KMeans(n_clusters=9, init='k-means++', max_iter=300, n_init=10,verbose=False)
        self.vectorizer = TfidfVectorizer(max_df=0.3, min_df=1, stop_words='english',use_idf=True,norm='l1')
        pass
    
    def get_path(self,year):
        return self.data_folder + year + '/' 
    
    def get_filenames(self,year):
        return sorted(os.listdir(self.data_folder+'/'+year+'/'))
    
    def read_all_speeches(self):
        self.corpus = {year:[] for year in self.years}
        for year in self.years:
            for speech in self.get_filenames(year):
                text = self.read_speech(speech,year).split()
                n=200
                chunks = [' '.join(text[i:i+n]) for i in range(0,len(text),n)]
                for chunk in chunks:
                    self.corpus[year].append(chunk)
#         self.corpus = {year:[] for year in self.years}
#         for year in self.years:
#             for speech in self.get_filenames(year):
#                 self.corpus[year].append(self.read_speech(speech,year))

    def read_speech(self, speech_filename,year):
        with open(self.data_folder +'/'+year+'/'+speech_filename, "rb") as file:
            text = file.read()
            if type(text) is not str:
                try:
                    text = text.decode("utf-8")
                except:
                    text = text.decode("iso-8859-1")
            text = text.replace("000", "")
            return text
        
    def cluster(self,year):
        X = self.vectorizer.fit_transform(dataloader.corpus[year])
        self.km.fit(X)
        order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()
        key_topics=list()
        for ind in order_centroids[0,:10]:
            key_topics.append(terms[ind])
            #print("%s " % terms[ind], end='')
        return key_topics


# In[116]:


years_list=[year[:9] for year in corpus_files]
presidents = [year[10:] for year in corpus_files]
df = pd.DataFrame(columns = years_list)
topics_by_year = {}


# In[117]:


for file_name,year in zip(corpus_files,years_list):
    dataloader=CorpusDataLoader()
    dataloader.read_all_speeches()
    key_topics=dataloader.cluster(file_name)
    topics_by_year[year]=key_topics
    df[year]=key_topics


# In[118]:


df=df.T
df['Presidents']=presidents


# # Bringing data together
# Putting everything into a table for a better visualization.

# In[119]:


def highlight(data, color='yellow'):
    return 'background-color: %s' % color

df.style.applymap(highlight,subset=['Presidents'])

