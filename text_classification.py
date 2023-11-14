
from datasets import load_dataset
import pandas as pd
imdb = dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#Load Dataset
imdb = load_dataset("imdb")
texts = imdb['test']['text']
labels = imdb['test']['label']

##Change the model to model of interest
model_path = "idistilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



accuracy = evaluate.load("accuracy")



def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

##Replace these with the labels for the dataset of interest
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}



model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2, id2label=id2label, label2id=label2id
)

##The hyper parameters can be changed according
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

##Save the model to huggingface
##Go to huggingface and create a model
#print("Saving last checkpoint of the model")
#huggingface_repo = "paultrust/zeply_irish_gov_pretrain"
#model.push_to_hub(huggingface_repo)
