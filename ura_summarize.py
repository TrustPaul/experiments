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

model_id = "tiiuae/falcon-40b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
        ],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"""
   Summarize the following text\n\n
    Text: {texts}\n\n###\n\nSummary: {labels}"""
    return example


#data = load_dataset("Abirate/english_quotes")
#data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
dataset = load_dataset("paultrust/ura_summarization")
train_data = dataset["train"]

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

dataset_blogs = Dataset.from_pandas(df_train)
dataset_split = dataset_blogs.train_test_split(test_size=0.00001, seed=1234)

data =  dataset_split.map(
  prepare_prompts
)
#data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["prompts"]), batched=True)

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=24,
        gradient_accumulation_steps=24,
        weight_decay=0.01,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        num_train_epochs=5,
        output_dir="outputs",
        report_to="none",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

print("Saving last checkpoint of the model")
huggingface_repo = "paultrust/ura_summary_falcon_40b"
model.push_to_hub(huggingface_repo)
    
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


model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model.eval()
for i in texts:
    with torch.cuda.amp.autocast():
        batch = tokenizer(i, return_tensors="pt")
        output_tokens = model.generate(**batch, max_new_tokens=20)

    summary =  tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    pred_summary.append( summary)
    
df_results_summary = pd.DataFrame()
df_results_summary['text'] = texts 
df_results_summary['human_summary'] = labels
df_results_summary['generated_summary'] = pred_summary

df_results_summary.to_csv('ura_summary.csv')


