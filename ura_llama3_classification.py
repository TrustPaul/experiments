import pandas as pd
from datasets import load_dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'

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

data = pd.read_csv("/home/ptrust/experiments/llm_data/agnews/agnews_train_40000.csv")
data = data.sample(n=20)
texts = data['text']
labels = data['label']


texts_used = []
labels_used = []
labels_predicted = []

question_schema = ResponseSchema(name="questions",
                             description="A list of 5 questions that can be answered from the text provided")
answers_schema = ResponseSchema(name="answers",
                                      description="The category where the text belongs, it must be World,Sports,Business,Science ")


response_schemas = [ answers_schema ]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()



for text, label in zip(texts, labels):


    try:
        review_template_2 = """\
        Categorize the provided text into one of the specified categories:
        - World
        - Sports
        - Business
        - Science
        Text: {text}

        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(template=review_template_2)

        system_prompt  = f"You are language model trained by openai that answers user questions"
        messages = prompt.format_messages(text=text, 
                                      format_instructions=format_instructions)

        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {  system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

        { messages }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Category:  """


        sequences = pipe(
            prompt,
            max_new_tokens=30,
            do_sample=True,
            top_k=5,
            return_full_text = False,
        )

    
        answers  = sequences[0]['generated_text']
        output_dict = output_parser.parse(answers)
        answers = output_dict.get('answers')
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

df_synethentic.to_excel('/home/ptrust/experiments/llm_data/agnews/agnews_llama3_70b_v2.xlsx')