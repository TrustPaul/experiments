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

model_name = 'EleutherAI/gpt-j-6b'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto'
)
model.eval()
print(f"Model loaded on {device}")


tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

from transformers import StoppingCriteria, StoppingCriteriaList



import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# gpt-j-6b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=50,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}"
)

llm = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=llm, prompt=prompt)


def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"""
   Summarize the following text in one paragraph
    Text: {texts} 
    Summary: """
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
df_train = df_train.sample(n=50)

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
    summary = llm_chain.predict(instruction=i).lstrip()
    print(summary)
    pred_summary.append( summary)
    
df_results_summary = pd.DataFrame()
df_results_summary['text'] = texts 
df_results_summary['human_summary'] = labels
df_results_summary['generated_summary'] = pred_summary

df_results_summary.to_csv('ura_summary.csv')
