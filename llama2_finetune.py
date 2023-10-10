from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import transformers
from datetime import datetime
from datasets import load_dataset

dataset = load_dataset("paultrust100/ura_summarization")
train_data = dataset["train"]

import pandas as pd
text_train = train_data['text']
label_train = train_data['label']

df_train = pd.DataFrame()
df_train['text'] = text_train
df_train['label'] =  label_train

train, test = train_test_split(df_train,test_size=0.1)


def create_fine_tuning_dataset(df):
    rows = []
    for i, row in df.iterrows():
      rows.append({"input":f"{row.text}", 'output':f"{row.label}"})



    return pd.DataFrame(rows)



df_json = create_fine_tuning_dataset(train)
test_json = create_fine_tuning_dataset(test)

df_json.to_json('/home/ptrust/experiments/hugging_data/train.jsonl', orient='records', lines=True)
test_json.to_json('/home/ptrust/experiments/hugging_data/test.jsonl', orient='records', lines=True)



train_dataset = load_dataset('json', data_files='/home/ptrust/experiments/hugging_data/train.jsonl', split='train')  
eval_dataset = load_dataset('json', data_files='/home/ptrust/experiments/hugging_data/test.jsonl', split='train')
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

def formatting_func(example):
    text = f"###  Summarize the following text\n\n Text: {example['input']}\n ### Summary: {example['output']}"
    return text

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token



max_length = 512 # This was an appropriate max length for my dataset

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt2)

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
    
print(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True
    
    

project = "journal-finetune"
base_model_name = "llama2-7b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=24,
        gradient_accumulation_steps=24,
        num_train_epochs=5,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        logging_dir="./logs",        # Directory for storing logs             # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="none",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"""
   "###  Summarize the following text\n\n Text: {texts}\n ### Summary: """
    return example
#data = load_dataset("Abirate/english_quotes")
#data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
dataset = load_dataset("paultrust100/ura_summarization")
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
    print(summary)
    pred_summary.append( summary)
    
df_results_summary = pd.DataFrame()
df_results_summary['text'] = texts 
df_results_summary['human_summary'] = labels
df_results_summary['generated_summary'] = pred_summary

df_results_summary.to_csv('/home/ptrust/experiments/hugging_data/ura_summary.csv')
