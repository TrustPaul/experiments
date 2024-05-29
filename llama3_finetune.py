import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
base_model_id = "HuggingFaceH4/zephyr-7b-beta"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

finetuned_model = "paultrust/ura_zephyr_summarize"
from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, finetuned_model)


from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("paultrust/ura_summarization")

texts = dataset['test']['text']
labels = dataset['test']['label']


texts_used = []
labels_used = []
labels_predicted = []


from transformers import pipeline
pipe = pipeline(
    model=ft_model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100,
)


from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("paultrust/ura_summarization")

texts = dataset['test']['text']
labels = dataset['test']['label']


texts_used = []
labels_used = []
labels_predicted = []



for text, label in zip(texts, labels):


    try:
        
        prompt = f"""
        Summarize the text provided in one short sentence
        Do not explain or give details, just give the summary
        Text: {text} 
        Summary  """



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

df_synethentic.to_excel('zephyr_7b_summary_finetune.xlsx')