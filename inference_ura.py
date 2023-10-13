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

model_id = "facebook/opt-350m"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

text = "Hello my name is"
device = "cuda:0"


def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"""
   Summarize the following text\n\n
    Text: {texts}\n\n###\n\nSummary: """
    return example
#data = load_dataset("Abirate/english_quotes")
#data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
dataset = load_dataset("paultrust/ura_summarization")
train_data = dataset["test"]

import pandas as pd
text_train = train_data['text']
label_train = train_data['label']

df_train = pd.DataFrame()
df_train['text'] = text_train
df_train['text_label'] =  label_train

texts = df_train['text'].tolist()
labels =  df_train['text_label'].tolist()
#df_train = sample_dataset( df_train, samples_per_class=100)
#df_train  =  df_train.sample(frac=1).reset_index(drop=True)

dataset_split = Dataset.from_pandas(df_train)
#dataset_split = dataset_blogs.train_test_split(test_size=0.00001, seed=1234)

data =  dataset_split.map(
  prepare_prompts
)
texts = data['prompts']
labels = data['text_label']
pred_summary = []

for i in texts:
    inputs = tokenizer(i, return_tensors="pt").to(device)
    outputs = model_4bit.generate(**inputs, max_new_tokens=20)
    summary =  tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(summary)
    pred_summary.append( summary)
    
df_results_summary = pd.DataFrame()
df_results_summary['text'] = texts 
df_results_summary['human_summary'] = labels
df_results_summary['generated_summary'] = pred_summary

df_results_summary.to_csv('ura_summary.csv')
