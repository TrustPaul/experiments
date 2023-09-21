import argparse
import os
from accelerate import Accelerator
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

"""
Fine-Tune LLMs on custom datasets on blog generation
We are using falcon model
You may be required to do a few changes for other models
Some changes could be the target modules in the lora config
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="tiiuae/falcon-7b-instruct")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=100, type=int)

    return parser.parse_args()

free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
max_memory = f"{free_in_GB-2}GB"

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
max_memory
print(max_memory)


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


def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"""
    Decide whether a Tweet's sentiment is positive, negative.
    Text: {texts}\n\n###\n\nAnswer: {labels}"""
    return example

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

def create_datasets(tokenizer, args):     
    dataset = load_dataset("imdb")
    train_data = dataset["train"]



    import pandas as pd
    text_train = train_data['text']
    label_train = train_data['label']

    df_train = pd.DataFrame()
    df_train['text'] = text_train
    df_train['label'] =  label_train
    df_train['text_label'] = df_train['label'].replace({0: 'negative', 1: 'positive'})

    texts = df_train['text'].tolist()
    labels =  df_train['text_label'].tolist()
    #df_train = sample_dataset( df_train, samples_per_class=100)
    #df_train  =  df_train.sample(frac=1).reset_index(drop=True)

    dataset_blogs = Dataset.from_pandas(df_train)
    dataset_split = dataset_blogs.train_test_split(test_size=0.00001, seed=args.seed)

    data =  dataset_split.map(
      prepare_prompts
)
    columns  = data["train"].features
    tokenizer.pad_token = tokenizer.eos_token
    data =  data["train"]
    train_dataset = data.map(lambda samples: tokenizer(samples["prompts"]), batched=True, remove_columns=columns)

    return train_dataset


def run_training(args, train_data, tokenizer):
    print("Loading the model")

    config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["query_key_value"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)


    print("Starting main loop")
    
    bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_8bit=True, device_map={"": Accelerator().process_index}, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
      args.model_path,
      quantization_config=bnb_config,
      device_map={"":0},
      trust_remote_code=True,
     )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    
    training_args = transformers.TrainingArguments(
      auto_find_batch_size=True,
      gradient_accumulation_steps=4,
      num_train_epochs=50,
      learning_rate=2e-5,
      fp16=True,
      save_total_limit=4,
      logging_steps=25,
      output_dir="./outputs",
      save_strategy='epoch',
      optim="paged_adamw_8bit",
      lr_scheduler_type = 'cosine',
      warmup_ratio = 0.05,
      report_to="wandb"
)

    trainer = transformers.Trainer(
         model=model,
         train_dataset=train_data,
         args=training_args,
         data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Training...")
    trainer.train()
    print("Saving last checkpoint of the model")
    huggingface_repo = "paultrust/epu_falcon_7b_instruct_4_bit"
    model.push_to_hub(huggingface_repo)
    
   # model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset= create_datasets(tokenizer, args)
    run_training(args, train_dataset,tokenizer)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llm model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
