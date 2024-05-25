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
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("paultrust/ura_summarization")



texts = dataset['train']['text']
labels = dataset['train']['label']

df_train = pd.DataFrame()
df_train['text'] = texts
df_train['label'] = labels

#df_train = df_train.sample(n=100)

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
    text = f"""
    <s>[INST] <<SYS>>
    You are a helpful langauge model designed to answer question faithfully
    <</SYS>>
    Summarize the text provided in one short sentence
    Do not explain or give details, just give the summary
    Text: {example['input']} [/INST]
    Summary: {example['output']}
        """

    return  text

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
    
    
output_dir = f"ura_finetuned_{base_model_id}"

tokenizer.pad_token = tokenizer.eos_token
huggingface_repo = "paultrust/zeply_irish_gov_pretrain"
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=48,
        gradient_accumulation_steps=48,
        num_train_epochs=5,
        learning_rate=2.5e-5,
        bf16=True,
        logging_dir="./logs",
        do_eval=True,           
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

print("Saving last checkpoint of the model")
huggingface_repo = "paultrust/ura_zephyr_summarize"
model.push_to_hub(huggingface_repo)
