from sklearn.datasets import load_digits
import numpy
import pandas as pd
from sklearn.cluster import KMeans
import os
import urllib.request as urllib2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from apricot import FeatureBasedSelection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from apricot import FacilityLocationSelection
from apricot import MaxCoverageSelection
from apricot import GraphCutSelection
from apricot import SaturatedCoverageSelection
from apricot import FeatureBasedSelection
from sklearn.cluster import KMeans




data = load_dataset('imdb')



x_train = data['train']['text']

y_train = data['train']['label']




train_df = pd.DataFrame({
    'text': x_train,
    'label':y_train
})

#test_df = pd.DataFrame({
#    'text':x_test_clean  ,
#    'label':y_test
#})


# In[ ]:


def sample_group(group):
    if len(group) > 1:
        return group.sample(n=1000)
    else:
        return group


# In[ ]:


#random_sample = train_df.groupby('label').apply(sample_group)

random_sample =  train_df
# In[ ]:


random_sample.shape


# In[ ]:


x_train = random_sample['text'].tolist()
y_train = random_sample['label'].tolist()


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-distilroberta-v1',device=device)
X_train_embedings = model.encode(x_train)
#X_test_embedings = model.encode(X_test)


# In[ ]:


embeddings= X_train_embedings
sentences = x_train


# In[2]:


n = len(embeddings)


# In[18]:


n


# In[19]:


# Create a mapping between the original sentences and their embeddings
mapping = {tuple(embedding.tolist()): sentence for sentence, embedding in zip(sentences, embeddings)}


# In[27]:


df_ = pd.DataFrame(X_train_embedings)
df_['label'] = y_train


# In[28]:


labels = train_df['label'].tolist()


# In[29]:


train_select = X_train_embedings
label_select = y_train




label_select =  np.array(y_train)
train_select = np.array(X_train_embedings)

def SubmodularSelection(x_train, y_train, k, submodular_selection, proportion):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
  cluster_labels=kmeans.labels_
  df = pd.DataFrame(x_train)
  df['label'] = cluster_labels
  X = {}
  Y = {}
  x_submodular = []
  y_submodular = []
  for i in range(k):
    d  = df[df['label']==i]
    index = d.index
    d = d.drop(columns=['label'])
    x_sub = np.array(d)
    n = round(len(x_sub)*proportion)
    x = np.array(d)
    y = y_train[index]
    if (submodular_selection == 'FACILITY') and (n > 0):
      X_subset, y_subset = FacilityLocationSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
    X[i] = X_subset
    Y[i] = y_subset
  for (k,x), (j,y) in zip(X.items(), Y.items()):
    x_submodular = x_submodular + x
    y_submodular = y_submodular + y


    if (submodular_selection == 'FEATURE') and (n > 0):
      X_subset, y_subset = FeatureBasedSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
      X[i] = X_subset
      Y[i] = y_subset
    for (i,x), (j,y) in zip(X.items(), Y.items()):
      x_submodular = x_submodular + x
      y_submodular = y_submodular + y


    if (submodular_selection == 'MAX-COVERAGE') and (n > 0):
      X_subset, y_subset = MaxCoverageSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
      X[i] = X_subset
      Y[i] = y_subset
    for (i,x), (j,y) in zip(X.items(), Y.items()):
      x_submodular = x_submodular + x
      y_submodular = y_submodular + y


    if (submodular_selection == 'SATU-COVERAGE') and (n > 0):
      X_subset, y_subset = SaturatedCoverageSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
      X[i] = X_subset
      Y[i] = y_subset
    for (i,x), (j,y) in zip(X.items(), Y.items()):
      x_submodular = x_submodular + x
      y_submodular = y_submodular + y


    if (submodular_selection == 'GRAPH-CUT') and (n > 0):
      X_subset, y_subset = GraphCutSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
      X[i] = X_subset
      Y[i] = y_subset
    for (i,x), (j,y) in zip(X.items(), Y.items()):
      x_submodular = x_submodular + x
      y_submodular = y_submodular + y
    return x_submodular, y_submodular


# In[35]:


def SubmodularSelection(x_train, y_train, k, submodular_selection, proportion):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
  cluster_labels=kmeans.labels_
  df = pd.DataFrame(x_train)
  #df['label'] = cluster_labels
  df['label'] = y_train
  X = {}
  Y = {}
  x_submodular = []
  y_submodular = []
  for i in df['label'].unique():
    d  = df[df['label']==i]
    index = d.index
    d = d.drop(columns=['label'])
    x_sub = np.array(d)
   # n = round(len(x_sub)*proportion)
    n =100
    x = np.array(d)
    y = y_train[index]
    if (submodular_selection == 'FACILITY') and (n > 0):
      X_subset, y_subset = FacilityLocationSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
  #x_submodular.append(X_subset)
  #y_submodular.append(y_subset)
    X[i] = X_subset
    Y[i] = y_subset
  for (k,x), (j,y) in zip(X.items(), Y.items()):
    x_submodular = x_submodular + x
    x_submodular.append(x)
    y_submodular = y_submodular + y
  return X, Y


# In[36]:


def SubmodularSelection(x_train, y_train, k, submodular_selection, proportion):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
  cluster_labels=kmeans.labels_
  df = pd.DataFrame(x_train)
  df['label'] = cluster_labels
  #df['label'] = y_train
  X = {}
  Y = {}
  x_submodular = []
  y_submodular = []
  for i in range(k):
    d  = df[df['label']==i]
    index = d.index
    d = d.drop(columns=['label'])
    x_sub = np.array(d)
   # n = round(len(x_sub)*proportion)
    n = proportion
    x = np.array(d)
    y = y_train[index]
    if (submodular_selection == 'FACILITY') and (n > 0):
      X_subset, y_subset = FacilityLocationSelection(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
  #x_submodular.append(X_subset)
  #y_submodular.append(y_subset)
    X[i] = X_subset
    Y[i] = y_subset
  for (k,x), (j,y) in zip(X.items(), Y.items()):
    x_submodular = x_submodular + x
    x_submodular.append(x)
    y_submodular = y_submodular + y
  return X, Y


# In[37]:


def SubmodularSelection(x_train, y_train, k, submodular_selection, proportion,sub_func):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
  cluster_labels=kmeans.labels_
  df = pd.DataFrame(x_train)
  df['label'] = cluster_labels
  #df['label'] = y_train
  X = {}
  Y = {}
  x_submodular = []
  y_submodular = []
  for i in df['label'].unique():
    d  = df[df['label']==i]
    index = d.index
    d = d.drop(columns=['label'])
    x_sub = np.array(d)
   # n = round(len(x_sub)*proportion)
    n =  proportion
    x = np.array(d)
    y = y_train[index]
    if n > 0:
      X_subset, y_subset = sub_func(n_samples= n, verbose=True).fit_transform(x, y)
      X_subset, y_subset  = X_subset.tolist(), y_subset.tolist()
  #x_submodular.append(X_subset)
  #y_submodular.append(y_subset)
    X[i] = X_subset
    Y[i] = y_subset
  return X, Y

cluster_number = 5
cluster_number = int(cluster_number)
function = 'FACILITY'
sub_func =  FacilityLocationSelection
proportion = 400
total = len(train_select)
number = int(0.1*total)
print(number)

X_subset, y_subset  = SubmodularSelection(x_train=train_select, y_train = label_select, k=cluster_number, submodular_selection = function , proportion = proportion,sub_func = sub_func)


x_submodular = []
y_submodular = []
for (k,x), (j,y) in zip(X_subset.items(), y_subset.items()):
  x_submodular.append(x)
  y_submodular.append(y)

from itertools import chain
x_submodular= list(chain.from_iterable(x_submodular))
y_submodular = list(chain.from_iterable(y_submodular))


X_selected_array = np.array(x_submodular)



# Look up the original sentence given an embedding
embedding_to_lookup = X_selected_array[2]
original_sentence = mapping[tuple(embedding_to_lookup.tolist())]
print(original_sentence)




x_selected_text = []
for i in X_selected_array:
  original_sentence = mapping[tuple(i.tolist())]
  x_selected_text.append(original_sentence)



df_selected_ = pd.DataFrame(
    {
        'text':x_selected_text,
     'label':y_submodular
    }
)



df_selected_['label'].unique()
df_selected_['label'].value_counts()



df_selected_.to_csv('/home/ptrust/experiments/data/')
