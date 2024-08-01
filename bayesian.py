from reliabots.icp import ConformalPredictor
from reliabots.ivap import IVAP
import reliabots.calibrutils as cu

import csv, codecs
from tqdm.notebook import tqdm
from pprint import pprint
import json, pickle
import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
from datasets import Features, Value, Sequence, ClassLabel, DatasetDict, Dataset
from datasets import load_dataset, load_metric, set_caching_enabled, concatenate_datasets
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AdamW
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
import plotly.express as px
import pandas as pd

import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import list_metrics
from datasets import load_metric
from transformers import AutoModel, AutoTokenizer
from transformers import *
from tqdm import tqdm
from sklearn.utils import class_weight
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm.auto import tqdm
import pandas as pd
import json

# Assuming the rest of the imports and definitions are already in place
import itertools

def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)

def predictive_entropy(mc_preds):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return entropy(np.mean(mc_preds, axis=0))

def mutual_information(mc_preds):
    """
    Compute the difference between the entropy of the mean of the
    predictive distribution and the mean of the entropy.
    """
    mutual_info = entropy(np.mean(mc_preds, axis=0)) - np.mean(entropy(mc_preds),
                                                               axis=0)
    return mutual_info

def get_rho(sigma, delta):
    """
    sigma is represented by softplus function  'sigma = log(1 + exp(rho))' to make sure it 
    remains always positive and non-transformed 'rho' gets updated during backprop.
    """
    rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
    return rho

def MOPED(model, det_model, det_checkpoint, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
 
    Example implementation for Bayesian model with variational layers.

    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo. Specifying Weight Priors in 
        Bayesian Deep Neural Networks with Empirical Bayes. Proceedings of the AAAI 
        Conference on Artificial Intelligence. AAAI 2020. 
        https://arxiv.org/abs/1906.05323
    """
    det_model.load_state_dict(torch.load(det_checkpoint))
    for (idx, layer), (det_idx,
                       det_layer) in zip(enumerate(model.modules()),
                                         enumerate(det_model.modules())):
        if (str(layer) == 'Conv1dReparameterization()'
                or str(layer) == 'Conv2dReparameterization()'
                or str(layer) == 'Conv3dReparameterization()'
                or str(layer) == 'ConvTranspose1dReparameterization()'
                or str(layer) == 'ConvTranspose2dReparameterization()'
                or str(layer) == 'ConvTranspose3dReparameterization()'
                or str(layer) == 'Conv1dFlipout()'
                or str(layer) == 'Conv2dFlipout()'
                or str(layer) == 'Conv3dFlipout()'
                or str(layer) == 'ConvTranspose1dFlipout()'
                or str(layer) == 'ConvTranspose2dFlipout()'
                or str(layer) == 'ConvTranspose3dFlipout()'):
            #set the priors
            layer.prior_weight_mu = det_layer.weight.data
            if layer.prior_bias_mu is not None:
               layer.prior_bias_mu = det_layer.bias.data

            #initialize surrogate posteriors
            layer.mu_kernel.data = det_layer.weight.data
            layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
            if layer.mu_bias is not None:
               layer.mu_bias.data = det_layer.bias.data
               layer.rho_bias.data = get_rho(det_layer.bias.data, delta)
        elif (str(layer) == 'LinearReparameterization()'
                or str(layer) == 'LinearFlipout()'):
            #set the priors
            layer.prior_weight_mu = det_layer.weight.data
            if layer.prior_bias_mu is not None:
               layer.prior_bias_mu.data = det_layer.bias

            #initialize the surrogate posteriors
            layer.mu_weight.data = det_layer.weight.data
            layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
            if layer.mu_bias is not None:
               layer.mu_bias.data = det_layer.bias.data
               layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

        elif str(layer).startswith('Batch'):
            #initialize parameters
            layer.weight.data = det_layer.weight.data
            if layer.bias is not None:
               layer.bias.data = det_layer.bias
            layer.running_mean.data = det_layer.running_mean.data
            layer.running_var.data = det_layer.running_var.data
            layer.num_batches_tracked.data = det_layer.num_batches_tracked.data

    model.state_dict()
    return model

class DatasetPrep(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred,average='weighted')

    return {"accuracy": accuracy, "f1": f1}

def run(train_path, test_path, model_path, MAXLEN, num_monte_carlo,prior_mu,prior_sigma,posterior_mu_init,posterior_rho_init,type_rep,moped_enable,moped_delta):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path )

    X_train, X_test, y_train, y_test = train_test_split(train_df['text'] , train_df['label'], test_size=0.2, random_state=42)

    train_df = pd.DataFrame({
        'text': X_train,
        'label': y_train
    })
    valid_df = pd.DataFrame({
        'text': X_test,
        'label': y_test
    })
    test_df = pd.DataFrame({
        'text': test_df['text'],
        'label': test_df['label']
    })

    train_df['clean_text'] = train_df['text']
    valid_df['clean_text'] = valid_df['text']
    test_df['clean_text'] = test_df['text']

    model_name = "roberta-large"
    max_length = MAXLEN

    num_labels = train_df['label'].nunique()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    train_texts = train_df['clean_text'].values.tolist()
    valid_texts = valid_df['clean_text'].values.tolist()
    test_texts = test_df['clean_text'].values.tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    train_labels = train_df['label'].values.tolist()
    valid_labels = valid_df['label'].values.tolist()
    test_labels = test_df['label'].values.tolist()

    train_dataset = DatasetPrep(train_encodings, train_labels)
    valid_dataset = DatasetPrep(valid_encodings, valid_labels)
    test_dataset = DatasetPrep(test_encodings, test_labels)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    eval_dataloader = DataLoader(valid_dataset, batch_size=32)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 100
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    const_bnn_prior_parameters = {
        "prior_mu": prior_mu,
        "prior_sigma": prior_sigma,
        "posterior_mu_init": posterior_mu_init,
        "posterior_rho_init": posterior_rho_init,
        "type": type_rep,
        "moped_enable": moped_enable,
        "moped_delta": moped_delta,
    }

    dnn_to_bnn(model, const_bnn_prior_parameters)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    batch_size = 10
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            ce_loss = outputs.loss
            kl = get_kl_loss(model)
            loss = ce_loss + kl / batch_size

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = load_metric("f1", trust_remote_code=True)
    model.eval()
    num_monte_carlo = num_monte_carlo
    output_mc = []
    preds = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output_mc = []
            outputs = model(**batch)
            for mc_run in range(num_monte_carlo):
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                output_mc.append(probs)
            output = torch.stack(output_mc)
            pred_mean = output.mean(dim=0)

            predictions = torch.argmax(pred_mean, axis=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            preds.append(predictions)
    result = metric.compute()

    f1 = result['f1']
    predictive_uncertainty = predictive_entropy(output.data.cpu().numpy())
    model_uncertainty = mutual_information(output.data.cpu().numpy())
    
    predictive_uncertainty = predictive_uncertainty.tolist()
    model_uncertainty =  model_uncertainty.tolist()
    
    return f1, predictive_uncertainty, model_uncertainty

import pandas as pd
import json

# Assuming the rest of the imports and definitions are already in place

# # Hyperparameters to iterate over
# dataset = ["causal", ]
# model_paths_list = ["google-bert/bert-base-uncased", "roberta-base"]
# monte_carlo_samples_list = [10, 100, 1000, 5000, 10000, 40000, 50000, 100000]
# prior_mu_list = [0.0, 0.2, 0.9]
# prior_sigma_list = [1, 0.1]
# type_rep_list = ["Flipout", "Reparameterization"]
# moped_enable_list = [True, False]
# moped_delta_list = [0, 0.5, 1]
# posterior_mu_init_list = [0, 0.1]  # Example values, adjust as needed
# posterior_rho_init_list = [-3, -2]  # Example values, adjust as needed


# Hyperparameters to iterate over
dataset = ["causal", "dailog"]
model_paths_list = ["google-bert/bert-base-uncased", "roberta-base"]
monte_carlo_samples_list = [50000, 100]
prior_mu_list = [0.0]
prior_sigma_list = [1]
type_rep_list = ["Flipout", "Reparameterization"]
moped_enable_list = [True, False]
moped_delta_list = [0]
posterior_mu_init_list = [0]  # Example values, adjust as needed
posterior_rho_init_list = [-3]  # Example values, adjust as needed



# Main execution
dataset_paths = {
    'causal': ('/home/ptrust/experiments/data/train_causal.csv', '/home/ptrust/experiments/data/test_causal.csv'),
    'emotion': ('/home/ptrust/experiments/data/train_emotion_sentiment.csv', '/home/ptrust/experiments/data/imdb_test.csv/test_emotion_sentiment.csv'),
    'dailog': ('/home/ptrust/experiments/data/daily_dialog_train.csv', '/home/ptrust/experiments/data/dailydialog_test.csv')
}
MAXLEN = 100
results = []

for dataset_name in dataset:
    train_path, test_path = dataset_paths[dataset_name]
    for model_path in model_paths_list:
        for num_monte_carlo in monte_carlo_samples_list:
            for prior_mu in prior_mu_list:
                for prior_sigma in prior_sigma_list:
                    for type_rep in type_rep_list:
                        for moped_enable in moped_enable_list:
                            for moped_delta in moped_delta_list:
                                for posterior_mu_init in posterior_mu_init_list:
                                    for posterior_rho_init in posterior_rho_init_list:
                                        f1, predictive_uncertainty, model_uncertainty = run(
                                            train_path, test_path, model_path, MAXLEN, num_monte_carlo,
                                            prior_mu, prior_sigma, posterior_mu_init, posterior_rho_init,
                                            type_rep, moped_enable, moped_delta
                                        )
                                        results.append({
                                            "dataset": dataset_name,
                                            "model_path": model_path,
                                            "num_monte_carlo": num_monte_carlo,
                                            "prior_mu": prior_mu,
                                            "prior_sigma": prior_sigma,
                                            "type_rep": type_rep,
                                            "moped_enable": moped_enable,
                                            "moped_delta": moped_delta,
                                            "posterior_mu_init": posterior_mu_init,
                                            "posterior_rho_init": posterior_rho_init,
                                            "f1": f1,
                                            "predictive_uncertainty": predictive_uncertainty,
                                            "model_uncertainty": model_uncertainty
                                        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save DataFrame to a CSV file
results_df.to_csv('/home/ptrust/experiments/imbalance/results_bayesian.csv', index=False)

# Optionally, also save as JSON
with open('/home/ptrust/experiments/imbalance/results_bayesian.json', 'w') as f:
    json.dump(results, f, indent=4)
