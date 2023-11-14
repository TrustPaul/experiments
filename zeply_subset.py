from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import transformers
from datetime import datetime
from datasets import load_dataset
import pandas as pd


#dataset = load_dataset('paultrust/epu_llm')

import pandas as pd
df = pd.read_csv('/notebooks/gpt/subsets/results/squad_random_examples_500.csv')

text_train = df['text'].tolist()
label_train = df['label'].tolist()


df_train = pd.DataFrame()
df_train['text'] = text_train
df_train['label'] =  label_train
#df_train['label_map'] =  label_train

# Define a mapping from numbers to labels
#mapping = {0: 'no', 1: 'yes'}

# Use the map function to replace numeric values with labels
#df_train['label'] = df_train['label_map'].map(mapping)

train, test = train_test_split(df_train,test_size=0.001)


def create_fine_tuning_dataset(df):
    rows = []
    for i, row in df.iterrows():
      rows.append({"input":f"{row.text}", 'output':f"{row.label}"})



    return pd.DataFrame(rows)



df_json = create_fine_tuning_dataset(train)
test_json = create_fine_tuning_dataset(test)

df_json.to_json('train.jsonl', orient='records', lines=True)
test_json.to_json('test.jsonl', orient='records', lines=True)



train_dataset = load_dataset('json', data_files='train.jsonl', split='train')  
eval_dataset = load_dataset('json', data_files='test.jsonl', split='train')
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

def formatting_func(example):
    text = f"""{example['input']}"""
    return text

def formatting_func(example):
    text = f"Text: {example['input']}"
    return text

base_model_id = "HuggingFaceH4/zephyr-7b-alpha"
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



max_length = 1024 # This was an appropriate max length for my dataset

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
huggingface_repo = "paultrust/squad_zeply_random_500"
run_name = huggingface_repo
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        num_train_epochs=5,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        logging_dir="./logs",        # Directory for storing logs             # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

print("Saving last checkpoint of the model")
huggingface_repo = "paultrust/squad_zeply_random_500"
model.push_to_hub(huggingface_repo)
