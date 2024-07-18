import itertools
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    XLNetTokenizer, XLNetForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    GPT2Tokenizer, GPT2ForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
import torch

# Download necessary NLTK resources
nltk.download('all')
nltk.download('punkt')
nltk.download('stopwords')

# Define paths and load the dataset
tweets_path = "https://github.com/PROVIA1/data/raw/main/cyberbullying_tweets.csv"
cyberbullying_tweets = pd.read_csv(tweets_path)
#cyberbullying_tweets = cyberbullying_tweets.sample(n=80)

# Drop missing values
cyberbullying_tweets = cyberbullying_tweets.dropna()

# Encode cyberbullying types
cyberbullying_type_mapping = {
    'age': 0,
    'ethnicity': 1,
    'gender': 2,
    'notcb': 3,
    'other': 4,
    'religion': 5
}
cyberbullying_tweets['cyberbullying_type'] = cyberbullying_tweets['cyberbullying_type'].map(cyberbullying_type_mapping)

# Preprocess the tweets
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [WordNetLemmatizer().lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

cyberbullying_tweets['filtered_tweet'] = cyberbullying_tweets['tweet'].astype(str).apply(preprocess_text)

# Split the data
train, test = train_test_split(cyberbullying_tweets, test_size=0.29)
x_train_texts = train['filtered_tweet'].tolist()
y_train = train['cyberbullying_type'].tolist()
x_test_texts = test['filtered_tweet'].tolist()
y_test = test['cyberbullying_type'].tolist()

# Create PyTorch dataset
class CyberbullyingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Tokenization function
def tokenize_texts(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Define model configurations
model_configs = {
    "BERT": {
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "pretrained_model_name": 'bert-base-uncased',
    },
    "DistilBERT": {
        "model_class": DistilBertForSequenceClassification,
        "tokenizer_class": DistilBertTokenizer,
        "pretrained_model_name": 'distilbert-base-uncased',
    },
    "XLNet": {
        "model_class": XLNetForSequenceClassification,
        "tokenizer_class": XLNetTokenizer,
        "pretrained_model_name": 'xlnet-base-cased',
    },
    "RoBERTa": {
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "pretrained_model_name": 'roberta-base',
    },
    "GPT-2": {
        "model_class": GPT2ForSequenceClassification,
        "tokenizer_class": GPT2Tokenizer,
        "pretrained_model_name": 'gpt2',
    }
}

# Define parameter configurations
num_train_epochs = [3, 5, 7]
learning_rates = [2e-5, 3e-5, 5e-5]
batch_sizes = [8, 16, 32]

# Create combinations of parameter configurations
param_combinations = list(itertools.product(num_train_epochs, learning_rates, batch_sizes))

# Define compute metrics function
metric = load_metric("accuracy", trust_remote_code=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Train and evaluate each model with different parameter configurations
results = []
for model_name, config in model_configs.items():
    tokenizer = config["tokenizer_class"].from_pretrained(config["pretrained_model_name"])
    
    # Add padding token if not present (specifically for GPT-2)
    if model_name == "GPT-2":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Set padding side to left
        model = config["model_class"].from_pretrained(config["pretrained_model_name"], num_labels=6)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model = config["model_class"].from_pretrained(config["pretrained_model_name"], num_labels=6)

    train_encodings = tokenize_texts(tokenizer, x_train_texts)
    test_encodings = tokenize_texts(tokenizer, x_test_texts)

    train_dataset = CyberbullyingDataset(train_encodings, y_train)
    test_dataset = CyberbullyingDataset(test_encodings, y_test)
    
    for (num_train_epochs, learning_rate, batch_size) in param_combinations:
        print(f"Training and evaluating {model_name} with epochs: {num_train_epochs}, lr: {learning_rate}, batch_size: {batch_size}...")
        
        training_args = TrainingArguments(
            output_dir=f'./results/{model_name}/{num_train_epochs}_{learning_rate}_{batch_size}',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",  # Changed to eval_strategy from evaluation_strategy
            save_strategy="epoch",     # Save checkpoints at the end of each epoch
            save_total_limit=1,        # Keep only the best checkpoint
            learning_rate=learning_rate,
            load_best_model_at_end=True  # Load the best model at end
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        
        # Collect the results
        results.append({
            "model": model_name,
            "learning_rate": learning_rate,
            "num_epochs": num_train_epochs,
            "batch_size": batch_size,
            "eval_loss": eval_results["eval_loss"],
            "eval_accuracy": eval_results["eval_accuracy"],
            "eval_precision": eval_results["eval_precision"],
            "eval_recall": eval_results["eval_recall"],
            "eval_f1": eval_results["eval_f1"],
            "eval_runtime": eval_results["eval_runtime"],
            "eval_samples_per_second": eval_results["eval_samples_per_second"],
            "eval_steps_per_second": eval_results["eval_steps_per_second"]
        })

        # Print evaluation results
        print(f"Results for {model_name} with epochs: {num_train_epochs}, lr: {learning_rate}, batch_size: {batch_size}: {eval_results}")

# Save results to a text file
with open("model_results.txt", "w") as f:
    for result in results:
        f.write(f"Model: {result['model']}, Learning Rate: {result['learning_rate']}, Num Epochs: {result['num_epochs']}, Batch Size: {result['batch_size']}\n")
        f.write(f"Eval Loss: {result['eval_loss']}\n")
        f.write(f"Eval Accuracy: {result['eval_accuracy']}\n")
        f.write(f"Eval Precision: {result['eval_precision']}\n")
        f.write(f"Eval Recall: {result['eval_recall']}\n")
        f.write(f"Eval F1 Score: {result['eval_f1']}\n")
        f.write(f"Eval Runtime: {result['eval_runtime']}\n")
        f.write(f"Eval Samples per Second: {result['eval_samples_per_second']}\n")
        f.write(f"Eval Steps per Second: {result['eval_steps_per_second']}\n")
        f.write("\n")
