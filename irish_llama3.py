import os
from huggingface_hub import InferenceClient
import os
from huggingface_hub import InferenceClient
import pandas as pd
from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import pandas as pd

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

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


data = pd.read_csv("/home/ptrust/experiments/llm_data/irish/data_irish.csv")

data = data.sample(n=10)
texts = data['texts']
questions = data['questions']
answers = data['answers']
urls = data['urls']


texts_used = []
questions_used = []
answers_used = []
urls_used = []
answers_predicted = []



for text, qn, ans, url in zip(texts, questions, answers, urls ):


    try:

        system_prompt  = f"You are language model trained by openai that answers user questions"
        user_msg_1 = f"""
            Given the text, answer the following question from the text provided. The answer should only be from the answer provided otherwise say no answer in the text
            Do not start your answer with according to the text, just generate the answer
        text: {text}
        Question: {qn} """

        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

        { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Answer:  """


        sequences = pipe(
            prompt,
            max_new_tokens=40,
            do_sample=True,
            top_k=5,
            return_full_text = False,
        )

    
        answers  = sequences[0]['generated_text']
        print(answers)
        answers_predicted.append(answers)



    except:
        answers = 'no summary'
        answers_predicted.append(answers)


    texts_used.append(text)
    questions_used.append(qn)
    answers_used.append(ans)
    urls_used.append( url)
    
    
    
    
df_synethentic = pd.DataFrame()
df_synethentic['text'] = texts_used
df_synethentic['question'] = questions_used
df_synethentic['chatgpt_answer'] =  answers_used
df_synethentic['llama3b'] = answers_predicted
df_synethentic['url'] =  urls_used

df_synethentic.to_csv('/home/ptrust/experiments/llm_data/irish/irish_train_llama3.csv')