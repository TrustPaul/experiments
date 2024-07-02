from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from ragatouille import RAGPretrainedModel
from langchain.retrievers import SVMRetriever
from langchain.retrievers import TFIDFRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load CSV data
df_full = pd.read_csv('/home/ptrust/experiments/llm_data/irish/data_irish.csv')

# df_full = pd.read_csv('/notebooks/gpt/ura/data_irish.csv')

df_full = df_full.sample(n=5)

# Extract lists from DataFrame
questions = df_full['questions'].tolist()  # Ensure these columns exist in df_full
answers = df_full['answers'].tolist()      # Ensure these columns exist in df_full
texts_full = df_full['texts'].tolist()

# Remove duplicates from the text lists
texts_full = list(set(texts_full))

# Combine text data for splitting
all_page_text_full = " ".join(texts_full)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits_full = text_splitter.split_text(all_page_text_full)

embedding_models = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-albert-small-v2",
    "sentence-transformers/all-distilroberta-v1",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/msmarco-distilbert-dot-v5"
]

# Function to format documents
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Function to run experiments on a set of texts
def run_experiments(texts, splits):
    experiment_results = []

    for model_name in embedding_models:
        # Initialize embeddings
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Initialize RAG model and vector database
        vectordb = Chroma.from_texts(texts, embedding=embeddings)

        # Initialize retrievers
        svm_retriever = SVMRetriever.from_texts(splits, embeddings)
        tfidf_retriever = TFIDFRetriever.from_texts(splits)

        for question, answer in zip(questions, answers):
            try:
                query = question

                # Experiment for different values of k
                for k in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                    docs = vectordb.similarity_search(query, k=k)
                    doc_chunks = [doc.page_content for doc in docs]
                    experiment_results.append({
                        'embedding_model': model_name,
                        'method': 'similarity_search',
                        'k': k,
                        'k_rerank': "no",
                        'question': question,
                        'answer': answer,
                        'results': doc_chunks
                    })

                    # If rerank
                    for k_rerank in [10, 15, 20, 25, 30]:
                        doc_chunks_to_rerank = vectordb.similarity_search(query, k=k_rerank)
                        doc_chunks_to_rerank = [doc.page_content for doc in doc_chunks_to_rerank]
                        relevant_docs = RERANKER.rerank(query, doc_chunks_to_rerank, k=4)
                        docs_rerank = [doc["content"] for doc in relevant_docs]
                        experiment_results.append({
                            'embedding_model': model_name,
                            'method': 'rerank',
                            'k': 4,
                            'k_rerank': k_rerank,
                            'question': question,
                            'answer': answer,
                            'results': docs_rerank
                        })

                # If max_marginal_relevance
                docs_mmr = vectordb.max_marginal_relevance_search(query, k=4, fetch_k=k)
                formatted_docs_mmr = [doc.page_content for doc in docs_mmr]
                experiment_results.append({
                    'embedding_model': model_name,
                    'method': 'max_marginal_relevance',
                    'k': 4,
                    'k_rerank': "no",
                    'question': question,
                    'answer': answer,
                    'results': formatted_docs_mmr
                })

                # If TFIDF retriever
                docs_tfidf = tfidf_retriever.get_relevant_documents(query, k=4)
                formatted_docs_tfidf = [doc.page_content for doc in docs_tfidf]
                experiment_results.append({
                    'embedding_model': model_name,
                    'method': 'TFIDF',
                    'k': 4,
                    'k_rerank': "no",
                    'question': question,
                    'answer': answer,
                    'results': formatted_docs_tfidf
                })

                # If SVM retriever
                docs_svm = svm_retriever.get_relevant_documents(query, k=4)
                formatted_docs_svm = [doc.page_content for doc in docs_svm]
                experiment_results.append({
                    'embedding_model': model_name,
                    'method': 'SVM',
                    'k': 4,
                    'k_rerank': "no",
                    'question': question,
                    'answer': answer,
                    'results': formatted_docs_svm
                })

            except Exception as e:
                experiment_results.append({
                    'embedding_model': model_name,
                    'method': 'error',
                    'k': 4,
                    'k_rerank': "no",
                    'question': question,
                    'answer': answer,
                    'results': f"Error: {str(e)}"
                })

    return experiment_results

# Run experiments on full texts
results_full = run_experiments(texts_full, splits_full)

# Create a DataFrame from the experiment results
df_results = pd.DataFrame(results_full)

# Save the results DataFrame to a new Excel file
df_results.to_excel('rag_experiments_results_full.xlsx', index=False)

# Define the system prompt
system_prompt = "You are a language model trained by OpenAI that answers user questions"

# Load the generation model
generation_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_id)
generation_model = AutoModelForCausalLM.from_pretrained(
    generation_model_id,
    load_in_4bit=True
)
generation_pipe = pipeline(
    model=generation_model,
    tokenizer=generation_tokenizer,
    task="text-generation",
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=150,
)

df_results = df_results.sample(n=10)
# Generate answers using the context
questions = df_results['question'].tolist()
answers = df_results['answer'].tolist()
contexts = df_results['results'].tolist()
generated_answers = []

for question, answer, context in zip(questions, answers, contexts):
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
        prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

            {  user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        sequences = generation_pipe(
            prompt,
            max_new_tokens=40,
            do_sample=True,
            top_k=5,
            return_full_text=False,
        )

        gen_answer = sequences[0]['generated_text']
        print(gen_answer)
        generated_answers.append(gen_answer)
    except Exception as e:
        print(f"Error: {e}")
        generated_answers.append("no answer")

df_results['generated_answer'] = generated_answers

# Load the evaluation model
evaluation_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
evaluation_tokenizer = AutoTokenizer.from_pretrained(evaluation_model_id)
evaluation_model = AutoModelForCausalLM.from_pretrained(
    evaluation_model_id,
    load_in_4bit=True
)
evaluation_pipe = pipeline(
    model=evaluation_model,
    tokenizer=evaluation_tokenizer,
    task="text-generation",
    temperature=0.1,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=40,
)

df_results = df_results.sample(n=10)
# Evaluate the generated answers
questions = df_results['question'].tolist()
answers = df_results['answer'].tolist()
generated_answers = df_results['generated_answer'].tolist()
evaluation_scores = []

for question, answer, gen_answer in zip(questions, answers, generated_answers):
    try:
        user_msg_2 =  f"""###Task Description:
            An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
            2.  write a score that is an integer between 1 and 5. You should refer to the score rubric.
            3. The output format should be in json looking as follows: 'score': an integer number between 1 and 5"
            4. Please do not generate any other opening, closing, and explanations. Just generate the score only

            ###The instruction to evaluate:
            {question}

            ###Response to evaluate:
            {gen_answer }

            ###Reference Answer:
            {answer }

            ###Score Rubrics:
            [Is the response correct, accurate, and factual based on the reference answer?]
            Score 1: The response is completely incorrect, inaccurate, and/or not factual.
            Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
            Score 3: The response is somewhat correct, accurate, and/or factual.
            Score 4: The response is mostly correct, accurate, and factual.
            Score 5: The response is completely correct, accurate, and factual.

            ###score:"""

        prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

            { user_msg_2 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        sequences = evaluation_pipe(
            prompt,
            max_new_tokens=25,
            do_sample=True,
            top_k=5,
            return_full_text=False,
        )

        eval_score = sequences[0]['generated_text']
        print(eval_score)
        evaluation_scores.append(eval_score)
    except Exception as e:
        print(f"Error: {e}")
        evaluation_scores.append("no score")

df_results['evaluation_score'] = evaluation_scores

# # Save the final DataFrame to a new Excel file
df_results.to_excel('/home/ptrust/experiments/llm_data/irish/rag_experiments_final_results.xlsx', index=False)

# # Save the results DataFrame to a new Excel file
# df_results.to_excel('rag_experiments_results_full.xlsx', index=False)
