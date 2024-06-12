import pandas as pd
import ast

data = pd.read_excel("/home/ptrust/experiments/hugging_data/gpt_generated_ireland_data.xlsx")

def str_to_list(s):
    try:
        value = ast.literal_eval(s)
        if isinstance(value, list):
            return value
        else:
            return s
    except (ValueError, SyntaxError):
        return s
    


# Applying the function to the relevant columns
data['answers'] = data['answers'].apply(str_to_list)
data['questions'] = data['questions'].apply(str_to_list)

data['length_qa'] = data['questions'].apply(len)
data['answer_qa'] = data['answers'].apply(len)

data = data[data['length_qa']==data['answer_qa']]

dfa = data.explode('answers')
dfq = data.explode('questions')

texts = dfa['text'].tolist() 
questions =  dfq['questions'].tolist()
answers = dfa['answers'].tolist() 
urls = dfa['url'].tolist() 

df_rag = pd.DataFrame()
df_rag['texts'] = texts
df_rag['questions'] = questions
df_rag['answers'] = answers
df_rag['urls'] = urls


df_rag.to_csv('/home/ptrust/experiments/llm_data/irish/data_irish.csv')