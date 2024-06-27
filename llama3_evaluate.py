import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        #attn_implementation="flash_attention_2", # if you have an ampere GPU
    )
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
    return pipe

def evaluate_model(questions, answers, generated_answers, model_type, pipe):
    scores = []
    for instruction, response, reference_answer in zip(questions, answers, generated_answers):
        try:
            user_msg_1 = f"""###Task Description:
            An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
            2.  write a score that is an integer between 1 and 5. You should refer to the score rubric.
            3. The output format should be in json looking as follows: 'score': an integer number between 1 and 5"
            4. Please do not generate any other opening, closing, and explanations. Just generate the score only

            ###The instruction to evaluate:
            {instruction}

            ###Response to evaluate:
            {response}

            ###Reference Answer:
            {reference_answer}

            ###Score Rubrics:
            [Is the response correct, accurate, and factual based on the reference answer?]
            Score 1: The response is completely incorrect, inaccurate, and/or not factual.
            Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
            Score 3: The response is somewhat correct, accurate, and/or factual.
            Score 4: The response is mostly correct, accurate, and factual.
            Score 5: The response is completely correct, accurate, and factual.

            ###score:"""

            system_prompt = "You are language model trained by openai that answers user questions"

            if model_type in ["meta-llama-8B", "meta-llama-70B"]:
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            elif model_type in ["mistralai-8x7B", "mistralai-7B", "llama-2-70B"]:
                prompt = f"""<s>[INST] <<SYS>>
                {system_prompt}
                <</SYS>>
                {user_msg_1} [/INST]
                """
            elif model_type in ["zephyr-7b", "nous-hermes-8x7B"]:
                prompt = f"""<|im_start|>system
                             {system_prompt}<|im_end|>
                            <|im_start|>user
                            {user_msg_1} <|im_end|>
                            <|im_start|>assistant"""

            sequences = pipe(
                prompt,
                max_new_tokens=40,
                do_sample=True,
                top_k=5,
                return_full_text=False,
            )

            answers = sequences[0]['generated_text']
            print(answers)
            scores.append(answers)
        except Exception as e:
            print(f"Error: {e}")
            answers = "no answer"
            scores.append(answers)

    return scores

def process_dataset(file_path, question_col, answer_col, gen_cols):
    df = pd.read_csv(file_path)
    df = df.sample(n=5)  # Adjust the sample size as needed

    questions = df[question_col].tolist()
    answers = df[answer_col].tolist()
    retrieved_docs = df['contexts'].tolist()

    # Load models
    pipes = {model_type: load_model(model_id) for model_type, model_id in model_configs.items()}

    # Evaluate each model with each set of generated answers
    all_scores = {model_type: {} for model_type in model_configs.keys()}
    for gen_col in gen_cols:
        generated_answers = df[gen_col].tolist()
        for model_type in model_configs.keys():
            pipe = pipes[model_type]
            all_scores[model_type][gen_col] = evaluate_model(questions, answers, generated_answers, model_type, pipe)

    # Add scores to the dataframe
    for model_type in model_configs.keys():
        for gen_col in gen_cols:
            df[f'{model_type}_score_{gen_col}'] = all_scores[model_type][gen_col]

    # Save the evaluation results to a new CSV file
    output_file = file_path.replace('.csv', '_evaluation_results.csv')
    df.to_csv(f"/home/ptrust/experiments/llm_data/irish/{output_file}", index=False)

# Define model configurations
model_configs = {
    "meta-llama-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama-70B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-2-70B": "meta-llama/Llama-2-70b-hf",
    "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    "nous-hermes-8x7B": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
}

# Process each dataset
process_dataset('/home/ptrust/experiments/llm_data/irish/rag_evaluation_irish_contexts.csv', 'question', 'true_answer', ['llama_7b_answer_x', 'llama_13b_answer', 'msitra_7b_instruct_answer'])
process_dataset('/home/ptrust/experiments/llm_data/irish/rag_evaluation_hse_contexts.csv', 'question', 'true_answer', ['llama_7b_answer', 'llama_13b_answer', 'mistra_7b_answer'])
process_dataset('/home/ptrust/experiments/llm_data/irish/statistical_graphs_with_context.csv', 'question', 'human_answer', ['gemini', 'llava_7b', 'llava_13b', 'gpt4answer'])
