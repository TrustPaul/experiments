import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load data
df = pd.read_csv('/home/ptrust/experiments/llm_data/irish/data_irish.csv')
df_1 =  pd.read_csv('/home/ptrust/experiments/llm_data/irish/irish_hse_docs.csv')
df_2 = pd.read_csv('/home/ptrust/experiments/llm_data/irish/statistical_graphs_with_context.csv')

df_1['texts'] = df_1['text'].tolist()
df_2['texts'] = df_2 ['text'].tolist()
# Define model configurations
model_configs = {
    "meta-llama-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    #"meta-llama-70B": "meta-llama/Meta-Llama-3-70B-Instruct",
   "mistralai-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
   "mistralai-7B": "mistralai/Mistral-7B-Instruct-v0.2",
     "mistralai-7B_v3": "mistralai/Mistral-7B-Instruct-v0.3",
   # "llama-2-70B": "meta-llama/Llama-2-70b-hf",
   # "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
   "nous-hermes-8x7B": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
}

# Sample data
df = df.sample(n=5)
df_1 = df_1.sample(n=5)
df_2 = df_2.sample(n=5)

def generate_questions_and_answers(model_type, model_id, df):
    texts = df['texts'].tolist()
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=20,
    )

    system_prompt = "You are a language model trained by OpenAI that answers user questions."

    questions_generated = []
    for context in texts:
        try:
            user_msg_1 = f"""
            Your task is to write a factoid question given a context.
            Your factoid question should be answerable with a specific, concise piece of factual information from the context.
            Your factoid question should be formulated in the same style as questions users could ask in a search engine.
            This means that your factoid question MUST NOT mention something like "according to the passage" or "context".
            You must also not start your answer with something like question, just generate the question
            You must only generate one question
            Now here is the context.

            Context: {context}\n
            Question:::"""

            if model_type in ["meta-llama-8B", "meta-llama-70B"]:
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif model_type in ["mistralai-8x7B", "mistralai-7B", "llama-2-70B","mistralai-7B_v3"]:
                prompt = f"""<s>[INST] <<SYS>>
                {system_prompt}
                <</SYS>>
                {user_msg_1} [/INST]
                """
            elif model_type in ["zephyr-7b", "nous-hermes-8x7B"]:
                prompt = f"""<|im_start|>system
                                 {system_prompt}<|im_end|>
                                <|im_start|>user
                                {user_msg_1} <|im_end|>"""

            sequences = pipe(
                prompt,
                max_new_tokens=40,
                do_sample=True,
                top_k=5,
                return_full_text=False,
            )

            answers = sequences[0]['generated_text']
            questions_generated.append(answers)
            print(answers)
        except:
            answers = "no question generated"
            questions_generated.append(answers)
            

    df[f'questions_generated_{model_type}'] = questions_generated

    answers_generated = []
    for context, question in zip(texts, questions_generated):
        try:
            user_msg_1 = f"""
            Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the number of the source document when relevant.
            If the answer cannot be deduced from the context, do not give an answer.
            You must MUST NOT mention something like "according to the passage" or "context" when giving the answer.
            You must also not start your answer with something like answer, just generate the answer
            Context:
            {context}
            ---
            Now here is the question you need to answer.

            Question: {question}

            Answer:::"""

            if model_type in ["meta-llama-8B", "meta-llama-70B"]:
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif model_type in ["mistralai-8x7B", "mistralai-7B", "llama-2-70B","mistralai-7B_v3"]:
                prompt = f"""<s>[INST] <<SYS>>
                {system_prompt}
                <</SYS>>
                {user_msg_1} [/INST]
                """
            elif model_type in ["zephyr-7b", "nous-hermes-8x7B"]:
                prompt = f"""<|im_start|>system
                                 {system_prompt}<|im_end|>
                                <|im_start|>user
                                {user_msg_1} <|im_end|>"""

            sequences = pipe(
                prompt,
                max_new_tokens=40,
                do_sample=True,
                top_k=5,
                return_full_text=False,
            )

            answers = sequences[0]['generated_text']
            answers_generated.append(answers)
            print(answers)
        except:
            answers = "no answer generated"
            answers_generated.append(answers)
            

    df[f'answers_generated_{model_type}'] = answers_generated

    question_groundedness_critique = []
    for context, question in zip(texts, answers_generated):
        try:
            user_msg_1 = f"""
            You will be given a context and a question 
            Your task is to provide a 'score' scoring how well one can answer the given question unambiguously with the given context.
            Give your answer on a scale of 1 to 10, where 1 means that the question is not answerable at all given the context, and 10 means that the question is clearly and unambiguously answerable with the context.

            Provide your answer as follows:

            You MUST provide values for 'score:' in your answer.
            The output format should be in json looking as follows: 'score': an integer number between 1 and 10"
            No explanations are need just provided the json answer score: your score

            Now here are the question and context.

            Question: {question}\n
            Context: {context}\n
            Answer::: """

            if model_type in ["meta-llama-8B", "meta-llama-70B"]:
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif model_type in ["mistralai-8x7B", "mistralai-7B", "llama-2-70B","mistralai-7B_v3"]:
                prompt = f"""<s>[INST] <<SYS>>
                {system_prompt}
                <</SYS>>
                {user_msg_1} [/INST]
                """
            elif model_type in ["zephyr-7b", "nous-hermes-8x7B"]:
                prompt = f"""<|im_start|>system
                                 {system_prompt}<|im_end|>
                                <|im_start|>user
                                {user_msg_1} <|im_end|>"""

            sequences = pipe(
                prompt,
                max_new_tokens=15,
                do_sample=True,
                top_k=5,
                return_full_text=False,
            )

            answers = sequences[0]['generated_text']
            question_groundedness_critique.append(answers)
            print(answers)
        except:
            answers = "no score"
            question_groundedness_critique.append(answers)
            
            

    df[f'question_groundedness_critique_{model_type}'] = question_groundedness_critique

    question_relevance_critique_critique = []
    for context, question in zip(texts, answers_generated):
        try:
            user_msg_1 = f"""
            You will be given a question.
            Your task is to provide a 'score' representing how useful this question can be to understand the irish government services and activities.
            Give your answer on a scale of 1 to 10, where 1 means that the question is not useful at all, and 10 means that the question is extremely useful.

            Provide your answer as follows:

            You MUST provide values for 'score:' in your answer.
            The output format should be in json looking as follows: 'score': an integer number between 1 and 10"
            No explanations are need just provided the json answer score: your score

            Question: {question}\n
            Answer::: """

            if model_type in ["meta-llama-8B", "meta-llama-70B"]:
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif model_type in ["mistralai-8x7B", "mistralai-7B", "llama-2-70B","mistralai-7B_v3"]:
                prompt = f"""<s>[INST] <<SYS>>
                {system_prompt}
                <</SYS>>
                {user_msg_1} [/INST]
                """
            elif model_type in ["zephyr-7b", "nous-hermes-8x7B"]:
                prompt = f"""<|im_start|>system
                                 {system_prompt}<|im_end|>
                                <|im_start|>user
                                {user_msg_1} <|im_end|>"""

            sequences = pipe(
                prompt,
                max_new_tokens=15,
                do_sample=True,
                top_k=5,
                return_full_text=False,
            )

            answers = sequences[0]['generated_text']
            question_relevance_critique_critique.append(answers)
            print(answers)
        except:
            answers  = "no score"
            question_relevance_critique_critique.append(answers)
            
            

    df[f'question_relevance_critique_critique_{model_type}'] = question_relevance_critique_critique

    question_standalone_critique_critique = []
    for context, question in zip(texts, answers_generated):
        try:
            user_msg_1 = f"""
            You will be given a question.
            Your task is to provide a 'score' representing how useful this question can be to understand the irish government services and activities.
            Give your answer on a scale of 1 to 10, where 1 means that the question is not useful at all, and 10 means that the question is extremely useful.

            Provide your answer as follows:

            You MUST provide values for 'score:' in your answer.
            The output format should be in json looking as follows: 'score': an integer number between 1 and 10"
            No explanations are need just provided the json answer score: your score

            Question: {question}\n
            Answer::: """

            if model_type in ["meta-llama-8B", "meta-llama-70B"]:
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif model_type in ["mistralai-8x7B", "mistralai-7B", "llama-2-70B","mistralai-7B_v3"]:
                prompt = f"""<s>[INST] <<SYS>>
                {system_prompt}
                <</SYS>>
                {user_msg_1} [/INST]
                """
            elif model_type in ["zephyr-7b", "nous-hermes-8x7B"]:
                prompt = f"""<|im_start|>system
                                 {system_prompt}<|im_end|>
                                <|im_start|>user
                                {user_msg_1} <|im_end|>"""

            sequences = pipe(
                prompt,
                max_new_tokens=15,
                do_sample=True,
                top_k=5,
                return_full_text=False,
            )

            answers = sequences[0]['generated_text']
            question_standalone_critique_critique.append(answers)
            print(answers)
        except:
            answers = "no score"
            question_standalone_critique_critique.append(answers)
            

    df[f'question_standalone_critique_critique_{model_type}'] = question_standalone_critique_critique

# Run each model sequentially on each dataframe
for model_type, model_id in model_configs.items():
    generate_questions_and_answers(model_type, model_id, df)
    generate_questions_and_answers(model_type, model_id, df_1)
    generate_questions_and_answers(model_type, model_id, df_2)

# Save the final dataframes
df.to_excel('irish_synthentic_data.xlsx', index=False)
df_1.to_excel('irish_hse_synthentic_data.xlsx', index=False)
df_2.to_excel('statistical_graph_synthentic_data.xlsx', index=False)
