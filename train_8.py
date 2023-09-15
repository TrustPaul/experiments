
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from datasets import Dataset, DatasetDict

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="imdb", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=1024, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

model_name_or_path = 'tiiuae/falcon-7b-instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if "gpt-neox" in model_name_or_path:
    model = prepare_model_for_int8_training(
        model, output_embedding_layer_name="embed_out", layer_norm_names=["layer_norm", "layernorm"]
    )
else:
    model = prepare_model_for_int8_training(model)

# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
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


target_modules = ["query_key_value"]  # workaround to use 8bit training on this model
config = LoraConfig(
    r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

block_size = 1024

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

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


from datasets import load_dataset

dataset = load_dataset("ag_news")

text = dataset['train']['text']
label = dataset['train']['label']

df = pd.DataFrame()
df['text'] = text
df['label'] = label

df['text_label'] = df['label'].replace({0: 'World', 1: 'Sports', 2:'Business', 3:'Science'})
df = sample_dataset(df, samples_per_class=10)
df  = df.sample(frac=1).reset_index(drop=True)

data = Dataset.from_pandas(df)



# Form prompts in the format mnli hypothesis: {hypothesis} premise: {premise} target: {class_label} <|endoftext|>
def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"""
    What is the topic for a given news headline? \n
              Chose from these  predefined topics \n\n
             - World \n
             - Sports \n
             - Bussiness \n
             - Science \n\n: {texts}\n\n###\n\nCategory: {labels}"""
    return example

data = data.map(
    prepare_prompts
)
columns  = data.features
data = data.map(lambda samples: tokenizer(samples["prompts"]), batched=True, remove_columns=columns)
data = data.map(group_texts, batched=True)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    num_train_epochs=10,
    report_to="wandb",
    run_name="falcon_7b_ag_news",
)

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=training_args
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

output_dir = "paultrust/falcon_7b_instruct_ag_news"

model.push_to_hub(output_dir)
