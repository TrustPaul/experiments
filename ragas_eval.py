import pandas as pd
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness,  context_precision,context_recall, context_entity_recall, answer_similarity
from datasets import Dataset
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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
from langchain.llms import HuggingFacePipeline
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import pandas as pd


import ast
def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except:
        return []

data = pd.read_csv("/home/ptrust/experiments/llm_data/irish/rag_evaluation_irish_contexts.csv")
data['contexts'] = data['contexts'].apply(convert_string_to_list)

# Preprocess Data
data = data[['question', 'true_answer', 'llama_7b_answer_x', 'llama_13b_answer', "msitra_7b_instruct_answer", "contexts"]]
data = data.dropna()

# Prepare dataset for evaluation
def prepare_dataset(data, model_column):
    dataset_dict = {
        "question": data['question'].tolist(),
        "contexts":data['contexts'].tolist(),
        "answer": data[model_column].tolist(),
        "ground_truth": data['true_answer'].tolist()
    }
    ds = Dataset.from_dict(dataset_dict)
    return ds

# Define metrics
metrics = [faithfulness, answer_relevancy, answer_correctness, context_precision,context_recall, context_entity_recall, answer_similarity]

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

# Embeddings
model_name = "sentence-transformers/paraphrase-albert-small-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

llm = HuggingFacePipeline(pipeline=pipe)

# Evaluate each model
def evaluate_model(data, model_column):
    dataset = prepare_dataset(data, model_column)
    scores = evaluate(dataset, metrics=metrics,llm=llm,embeddings=embeddings)
    return scores

# Evaluate models
llama_7b_evaluation = evaluate_model(data, 'llama_7b_answer_x')
llama_13b_evaluation = evaluate_model(data, 'llama_13b_answer')
mistra_7b_evaluation = evaluate_model(data, 'msitra_7b_instruct_answer')



# Display Evaluation Results
print("llama_7b_evaluation:", llama_7b_evaluation)
print("llama_13b_evaluation:", llama_13b_evaluation)
print("mistra_7b_evaluation:", mistra_7b_evaluation)

# Convert evaluation results to DataFrame
llama_7b_evaluation_df = pd.DataFrame([llama_7b_evaluation])
llama_13b_evaluation_df = pd.DataFrame([llama_13b_evaluation])
mistra_7b_evaluation_df = pd.DataFrame([mistra_7b_evaluation])

llama_7b_evaluation_df.to_csv("/home/ptrust/experiments/llm_data/irish/llama_7b_evaluation.csv")
llama_13b_evaluation_df.to_csv("/home/ptrust/experiments/llm_data/irish/llama_13b_evaluation.csv")
mistra_7b_evaluation.to_csv("/home/ptrust/experiments/llm_data/irish/mistra_7b_evaluation.csv")