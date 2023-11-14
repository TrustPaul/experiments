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
answers = []
for i in answered:
    answer_extract = i['text']
    answers.append(answer_extract)
    
df_train = pd.DataFrame()
df_train['question'] = question
df_train['content'] = contexts
df_train['answered'] = answers 
df_train['answer'] = df_train['answered'].apply(lambda x: ', '.join(map(str, x)))

data = Dataset.from_pandas(df_train)

system_prompt = """You are a language model that answers user questions based on the provided instructions
"""

user_prompt = """
Answer the provided question from the provided text
This is an extractive question answering task, so you must extract the answer from text, rather than generating your own answer
"""
E_INST = "</s>"
user, assistant = "<|user|>", "<|assistant|>"

prompt = f"{system_prompt}{E_INST}\n{user}\n{user_prompt.strip()}{E_INST}"
def add_prefix(example):
    example["prompt"] = f"""
    {prompt}
    Question: {example['question']}\n 
    Text: {example['content']}
    \n{assistant}\n {example['answer']}"""
    return example




tokenized_train_dataset = data.map(add_prefix)
data = tokenized_train_dataset


x_train = data['prompt']

X_test = data['prompt']

y_train = data['answer']
y_test = data['answer']




train_df = pd.DataFrame({
    'text': x_train,
    'label':y_train
})


df_selected_= train_df.sample(n=500)
print(df_selected_.head())




df_selected_.to_csv('/notebooks/gpt/subsets/results/squad_random_examples_500.csv')
print(df_selected_.shape)