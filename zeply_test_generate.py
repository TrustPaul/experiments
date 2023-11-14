import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from datasets import load_dataset

from datasets import load_dataset
import pandas as pd

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
df_train['answer'] = answers

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
    \n{assistant}\n """
    return example




tokenized_train_dataset = data.map(add_prefix)
data = tokenized_train_dataset


x_train = data['prompt']

X_test = data['prompt']

y_train = data['answer']
y_test = data['answer']

        
        
    

#print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=True))
# model.save('./test.pt')

df_pred = pd.DataFrame()
df_pred['text'] = X_test
df_pred['true_labels'] = y_test
df_pred = df_pred.sample(n=1000)
print(df_pred['text'][0])

df_pred.to_excel('/notebooks/gpt/subsets/results/squad_test_1000.xlsx')
