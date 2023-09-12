
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

model_name_or_path = 'EleutherAI/gpt-j-6b'
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=True,
    device_map="auto",
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

target_modules = None
if "gpt-neox" in model_name_or_path:
    target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on this model
config = LoraConfig(
    r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

block_size = 1024

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

data = Dataset.from_pandas(df)



# Form prompts in the format mnli hypothesis: {hypothesis} premise: {premise} target: {class_label} <|endoftext|>
def prepare_prompts(example):
    texts = example["text"]
    labels = example["text_label"]

    example[
        "prompts"
    ] = f"Your provided with new article, chose what category it belongs to from this list [World, Sports, Bussiness, Science]: {texts}\n\n###\n\nCategory: {labels}"
    return example

data = data.map(
    prepare_prompts
)
columns  = data.features
data = data.map(lambda samples: tokenizer(samples["prompts"]), batched=True, remove_columns=columns)
data = data.map(group_texts, batched=True)

model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

output_dir = "paultrust/ag_news_version_one_gpt_j_6b"

model.push_to_hub(output_dir)
