import os
from huggingface_hub import InferenceClient
import os
from huggingface_hub import InferenceClient
import pandas as pd
from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import pipeline
pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
)


from datasets import load_dataset


from datasets import load_dataset

data = pd.read_csv("/home/ptrust/experiments/llm_data/agnews/agnews_train_40000.csv")

texts = data['text']
labels = data['label']


texts_used = []
labels_used = []
labels_predicted = []



for text, label in zip(texts, labels):


    try:
        
        text_ = f"""
      Categorize the provided text into one of the specified categories:
        - World
        - Sports
        - Business
        - Science
        Text: {text} \n\n
        The category must one of the following ['World', 'Sports', 'Business', 'Science']
        Return output as json format 'category': category   """

        prompt = f"""<|im_start|>system
              You are a helpful langauge model designed to answer question faithfully<|im_end|>
            <|im_start|>user
            {text_}<|im_end|>
            <|im_start|>assistant"""


        sequences = pipe(
            prompt,
            max_new_tokens=15,
            do_sample=True,
            top_k=5,
            return_full_text = False,
        )

    
        answers  = sequences[0]['generated_text']
        print(answers)
        labels_predicted.append(answers)



    except:
        answers = 'no summary'
        labels_predicted.append(answers)


    texts_used.append(text)
    labels_used.append(label)
    
    
    
    
df_synethentic = pd.DataFrame()
df_synethentic['text'] = texts_used
df_synethentic['summary'] = labels_used
df_synethentic['pred_summary'] = labels_predicted

df_synethentic.to_csv('/home/ptrust/experiments/llm_data/agnews/agnews_mistra_45b_dpo.csv')