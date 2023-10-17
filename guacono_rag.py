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


model_name = "huggyllama/llama-7b"
adapters_name = 'timdettmers/guanaco-7b'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory= {i: '24000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)
model = PeftModel.from_pretrained(model, adapters_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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



qn_answered= []
text_chunks_used = []
answer_generated = []
true_answer = []

for qn, con in zip(questions_train,answer_train):
    context = get_text_docs(qn)
    prompt_qn = f"""
    Answer the question from the given context:
    Question: {qn}\n\n
    Context: {context}\n\n
    Answer: 
    """

    formatted_prompt = (
        f"A chat between a curious human and an artificial intelligence assistant."
        f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        f"### Human: {prompt_qn} ### Assistant:"
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=50)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pattern = r'### Assistant: (.*?)(?:### Human:|$)'

    matches = re.findall(pattern, output, re.DOTALL)
    generated_questions = []
    for match in matches:

        qn_extract = match.strip()
        generated_questions.append(qn_extract)

    inputstring = generated_questions[0]
    print(inputstring)
    answer_generated.append(inputstring)
    qn_answered.append(qn)
    text_chunks_used.append( context)
    true_answer.append(con)
    
    
df_train = pd.DataFrame()
df_train['question'] = qn_answered
df_train['answer'] = answer_generated 
df_train['true_answer'] = answer_generated

#expanded_df = df_train.explode('questions')
df_train.to_csv('irish_guacono_7b_rag_12000.csv')
#expanded_df.to_csv('exploded_prompt_dataset_questions.csv')