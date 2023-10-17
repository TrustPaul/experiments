import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from torch import cuda, bfloat16
import transformers
from datasets import load_dataset, Dataset
from langchain.embeddings import (
    HuggingFaceEmbeddings, 
    SentenceTransformerEmbeddings
)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Weaviate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DataFrameLoader
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import re
from langchain.vectorstores import Chroma

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")




tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
    


generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=50,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
   


from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline



dataset = load_dataset("paultrust/gpt_geneted_irish_dataset_1200")
train_data = dataset["train"]
texts_train = train_data['text']
questions_train = train_data['question']
answer_train = train_data['answer']
url_train = train_data['url']

df  = pd.DataFrame()
df['text'] = texts_train

import pandas as pd
#df = pd.read_csv('gen_questions_datasets.csv')
#df = df.sample(n=10)
#data = df.loc[(df['ans_len'] >= 200) & (df['ans_len'] <= 2048 )]
data = df.sample(n=10)
texts = data['text'].tolist()

loader = DataFrameLoader(data, page_content_column="text")

documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

chunk_size =2000
chunk_overlap = 200
def get_text_splits(text_chunk):

  textSplit = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function = len,
    separators=[ ""]
)
  docs = text_splitter.split_documents(documents)
  return  docs
docs = split_docs(documents)
#print(len(docs))

#testing out the above function with the open source 
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-albert-small-v2")

db = Chroma.from_documents(docs, embeddings)


def get_text_docs(query):
    matching_docs = db.similarity_search(query)
    all_page_text=[p.page_content for p in matching_docs]
    joined_page_text="\n\n".join(all_page_text)
    #all_page_source=[d.metadata['url'] for d in matching_docs]
    #joined_page_sources="\n\n".join(all_page_source)
    
    return joined_page_text

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["text",'question'],
    template="""
    Answer the following using the following text\n\n
    Question: {question}\n\n
    Text: {text}\n\n
    Answer:  """
)

llm = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=llm, prompt=prompt)

qn_answered= []
text_chunks_used = []
answer_generated = []
true_answer = []

for i, j  in zip(texts_train,questions_train):
    context = get_text_docs(i)
    summary = llm_chain.predict(text=context, question = j).lstrip()
    print(summary)
    answer_generated.append( summary)
    qn_answered.append(i)
    
    
df_train = pd.DataFrame()
df_train['question'] = qn_answered
df_train['answer'] = answer_generated 
df_train['true_answer'] = answer_train
df_train['text'] = texts_train
df_train['url'] = url_train

#expanded_df = df_train.explode('questions')
df_train.to_csv('irish_falcon_7b_instruct_rag_12000.csv')
