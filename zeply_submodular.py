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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import OPTICS



from datasets import Dataset





from datasets import load_dataset

dataset = load_dataset("squad")

contexts = dataset['train']['context']
question = dataset['train']['question']
answered = dataset['train']['answers']
answers = []
for i in answered:
    answer_extract = i['text']
    answers.append(answer_extract)
    
df_train = pd.DataFrame()
df_train['question'] = question
df_train['content'] = question
df_train['answer'] = question

data = Dataset.from_pandas(df_train)

system_prompt = """You are a language model that answers user questions based on the provided instructions
"""

user_prompt = """
Answer the provided question from the provided context.
This task is an extractive question answering task and your answer must short and extracted from the text
"""
E_INST = "</s>"
user, assistant = "<|user|>", "<|assistant|>"

prompt = f"{system_prompt}{E_INST}\n{user}\n{user_prompt.strip()}{E_INST}"
def add_prefix(example):
    example["prompt"] = f"""
    {prompt}
    Question: {example['question']}\n 
    Content: {example['content']}
    \n{assistant}\n  {example['answer']}"""
    return example

tokenized_train_dataset = data.map(add_prefix)
data = tokenized_train_dataset


x_train = data['train']['prompt']

X_test = data['test']['prompt']

y_train = data['train']['answer']
y_test = data['test']['answer']
y_train  = np.array(y_train)
y_test  = np.array(y_test)
x_train  = np.array(x_train)


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


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = SentenceTransformer('all-mpnet-base-v2',device=device)
#X_train_embedings = model.encode(x_train)
#X_test_embedings = model.encode(X_test)


# In[ ]:

#vectorizer = CountVectorizer(max_features=5000)
#from sklearn.feature_extraction.text import TfidfTransformer

#vectorizer = CountVectorizer(max_features=5000)
#from sklearn.feature_extraction.text import TfidfTransformer

#X_train_embedings = vectorizer.fit_transform(x_train)

#X_train_embedings = X_train_embedings.toarray()


vectorizer = CountVectorizer(max_features=2000)
X_train_embedings = vectorizer.fit_transform(x_train)

X_train_embedings = X_train_embedings.toarray()

embeddings= X_train_embedings
sentences = x_train


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



def SubmodularSelection(x_train, y_train, k, submodular_selection, proportion,sub_func):
  kmeans = KMeans(n_clusters=k, random_state=0).fit(x_train)
  #kmeans = OPTICS(min_samples=10, metric='cosine').fit(x_train)
  cluster_labels=kmeans.labels_
  df = pd.DataFrame(x_train)
  df['label'] = cluster_labels
  #df['label'] = y_train
  X = {}
  Y = {}
  x_submodular = []
  y_submodular = []
  total_size = df.shape[0]
  for i in df['label'].unique():
    d  = df[df['label']==i]
    index = d.index
    d = d.drop(columns=['label'])
    x_sub = np.array(d)
   # n = round(len(x_sub)*proportion)
    n =  proportion
    
    fraction = len(x_sub)/total_size
    n = round(fraction* proportion)
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

cluster_number = 2
cluster_number = int(cluster_number)
function = 'FACILITY'
#sub_func =  FacilityLocationSelection
sub_func =   MaxCoverageSelection
proportion = 500
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



df_selected_.to_csv('/notebooks/gpt/subsets/results/qnli_max_coverage_cluster_2_examples_500.csv')
print(df_selected_.shape)