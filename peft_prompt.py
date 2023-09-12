from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
model_name_or_path = "facebook/opt-350m"
tokenizer_name_or_path = "facebook/opt-350m"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
text_column = "text"
label_column = "label"
max_length = 200
lr = 3e-2
num_epochs = 5
batch_size = 32


# In[18]:


import pandas as pd
from datasets import load_dataset, Dataset

def sample_dataset(df_model, samples_per_class=2):
    # Define the number of samples you want to randomly select from each class

    # Create an empty DataFrame to store the sampled results
    sampled_df = pd.DataFrame(columns=df_model.columns)

    # Iterate through each class, randomly sample from it, and append the samples to the sampled_df
    for class_label in df_model['label'].unique():
        class_data = df_model[df_model['label'] == class_label]
        random_samples = class_data.sample(n=samples_per_class, random_state=42)
        sampled_df = pd.concat([sampled_df, random_samples])
    return  sampled_df


dataset_name = "ag_news"
dataset = load_dataset(
    dataset_name,
    split='train'
)

dataset = dataset.train_test_split(test_size=0.1, seed=None)
train_data = dataset["train"]
valid_data = dataset["test"]
print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")


text_train = train_data['text']
label_train = train_data['label']

df_train = pd.DataFrame()
df_train['text'] = text_train 
df_train['label'] =  label_train

df_train['text_label'] = df_train['label'].replace({0: 'World', 1: 'Sports', 2:'Business', 3:'Science'})
texts = df_train['text'].tolist()
labels =  df_train['text_label'] .tolist()
prompts = []
for i, j in zip(texts, labels):
    prompt = f""" What is the topic for a given news headline? \n
              Chose from these  predefined topics \n\n
             - World \n
             - Sports \n
             - Bussiness \n
             - Science \n\n

             Text: {i}\n\n###\n\n """
    prompts.append(prompt)

df_model = pd.DataFrame()

df_model['text'] = prompts
df_model['label'] = labels

df_model = sample_dataset(df_model, samples_per_class=5000)
df_model  = df_model.sample(frac=1).reset_index(drop=True)

dataset = Dataset.from_pandas(df_model)
dataset = dataset.train_test_split(test_size=0.01, seed=None)




dataset['train']['label'][0]

classes = dataset['train']['label']
classes= list(set(classes))


# data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)


def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["train"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)



def test_preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    model_inputs = tokenizer(inputs)
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    return model_inputs


test_dataset = dataset["test"].map(
    test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


# creating model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()




# model
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)




# training and evaluation
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        #         print(batch)
        #         print(batch["input_ids"].shape)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


# In[32]:


model.to(device)
model.eval()


# In[36]:


dataset_name = "ag_news"
dataset = load_dataset(
    dataset_name,
    split='test'
)

dataset = dataset.train_test_split(test_size=0.99, seed=None)
train_data = dataset["train"]
valid_data = dataset["test"]
print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")


text_train = train_data['text']
label_train = train_data['label']

df_train = pd.DataFrame()
df_train['text'] = text_train 
df_train['label'] =  label_train

df_train['text_label'] = df_train['label'].replace({0: 'World', 1: 'Sports', 2:'Business', 3:'Science'})
texts = df_train['text'].tolist()
labels =  df_train['text_label'] .tolist()
prompts = []
for i, j in zip(texts, labels):
    prompt = f""" What is the topic for a given news headline? \n
              Chose from these  predefined topics \n\n
             - World \n
             - Sports \n
             - Bussiness \n
             - Science \n\n

             Text: {i}\n\n###\n\n """
    prompts.append(prompt)

df_model = pd.DataFrame()

df_model['text'] = prompts
df_model['label'] = labels

dataset = Dataset.from_pandas(df_model)







texts = df_model['text'].tolist()
labels = df_model['label'].tolist()




pred_labels = []
true_labels = []

for text, label in zip(texts, labels):
    inputs = tokenizer(f'{text_column} : {text} Label : ', return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=1, eos_token_id=3
        )
        pred_label = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        pred_labels.append(pred_label)
        true_labels.append(label)


df_results = pd.DataFrame()
df_results['pred'] = pred_labels
df_results['label'] = true_labels
df_results.to_csv('/home/ptrust/experiments/data/results_peft_opt.csv')
# saving model
peft_model_id = "paultrust/ag_news_peft"
tokenizer.push_to_hub(peft_model_id)
model.push_to_hub(peft_model_id)










