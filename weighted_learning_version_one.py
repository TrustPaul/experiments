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
import torch
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from transformers import Trainer
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import list_metrics
from datasets import load_metric
from transformers import AutoModel, AutoTokenizer 
from transformers import *
from tqdm import tqdm
from sklearn.utils import class_weight
#import texthero as hero
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
#Maximum Margin Loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#Data
import pandas as pd
from torch.optim import AdamW  
import torch
import torch.nn as nn
import torch
from transformers import Trainer
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

def sample_under_over(train, strategy):
    # Extract features and labels
    X_train = train[['text']]
    y_train = train['label']
    
    if strategy == "under":
    
        # Apply undersampling
        rus = RandomUnderSampler(random_state=42)
        X_train_undersampled, y_train_undersampled = rus.fit_resample(X_train, y_train)
        train_undersampled_df = pd.concat([X_train_undersampled, y_train_undersampled], axis=1)
        return train_undersampled_df
    elif strategy == "over":
        # Apply oversampling
        ros = RandomOverSampler(random_state=42)
        X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train, y_train)
        train_oversampled_df = pd.concat([X_train_oversampled, y_train_oversampled], axis=1)
        return train_oversampled_df

    
class WeightedTrainer(Trainer):
    def __init__(self,class_weighting, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weighting  =class_weighting
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weighting)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


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


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(
            self.s * output.to("cuda"), target.to("cuda"), weight=self.weight.to("cuda")
        )
    
class HMMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, gamma=1.1, ldam=False):
        super(HMMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.max_m = max_m
        self.gamma = gamma
        self.ldam = ldam

    def weight(self, freq_bias, target, args):

        index = torch.zeros_like(freq_bias, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        # plus 1 affects top-1 acc.
        cls_num_list = (index_float.sum(0).data.cpu() + 1)

        beta = args.beta

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to('cuda')

        return per_cls_weights

    def obj_margins(self, rm_obj_dists, labels, index_float, max_m):

        obj_neg_labels = 1.0 - index_float
        obj_neg_dists = rm_obj_dists * obj_neg_labels

        min_pos_prob = rm_obj_dists[:, labels.data.cpu().numpy()[0]].data
        max_neg_prob = obj_neg_dists.max(1)[0].data

        # estimate the margin between dists and gt labels
        batch_m_fg = torch.max(
            min_pos_prob - max_neg_prob,
            torch.zeros_like(min_pos_prob))[:,None]

        mask_fg = (batch_m_fg > 0).float()
        batch_fg = torch.exp(-batch_m_fg - max_m * self.gamma) * mask_fg

        batch_m_bg = torch.max(
            max_neg_prob - min_pos_prob,
            torch.zeros_like(max_neg_prob))[:,None]

        mask_ng = (batch_m_bg > 0).float()
        batch_ng = torch.exp(-batch_m_bg - max_m) * mask_ng
        batch_m = batch_ng + batch_fg
        return batch_m.data

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))

        # 1.0 - [0.5] => [0.0 ~ 0.5]
        if self.ldam :
            max_m = self.max_m - batch_m
        else:
            max_m = self.max_m

        with torch.no_grad():
            batch_hmm = self.obj_margins(x, target, index_float, max_m)

        x_m = x - batch_hmm

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output.to('cuda'), target.to('cuda'), weight=self.weight.to('cuda'))



class DROLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, class_weights=None, epsilons=None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.class_weights = class_weights
        self.epsilons = epsilons

    def pairwise_euaclidean_distance(self, x, y):
        return torch.cdist(x, y)

    def pairwise_cosine_sim(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        return torch.matmul(x, y.T)

    def forward(self, batch_feats, batch_targets, centroid_feats=None, centroid_targets=None):
        device = (torch.device('cuda'))

        classes, positive_counts = torch.unique(batch_targets, return_counts=True)
        centroid_classes = torch.unique(centroid_targets)
        train_prototypes = torch.stack([centroid_feats[torch.where(centroid_targets == c)[0]].mean(0)
                                        for c in centroid_classes])
        pairwise = -1 * self.pairwise_euaclidean_distance(train_prototypes, batch_feats)

        # epsilons
        if self.epsilons is not None:
            mask = torch.eq(centroid_classes.contiguous().view(-1, 1), batch_targets.contiguous().view(-1, 1).T).to(
                device)
            a = pairwise.clone()
            pairwise[mask] = a[mask] - self.epsilons[batch_targets].to(device)

        logits = torch.div(pairwise, self.temperature)

        # compute log_prob
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        log_prob = torch.stack([log_prob[:, torch.where(batch_targets == c)[0]].mean(1) for c in classes], dim=1)

        # compute mean of log-likelihood over positive
        mask = torch.eq(centroid_classes.contiguous().view(-1, 1), classes.contiguous().view(-1, 1).T).float().to(
            device)
        log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob_pos
        # weight by class weight
        if self.class_weights is not None:
            weights = self.class_weights[centroid_classes]
            weighted_loss = loss * weights
            loss = weighted_loss.sum() / weights.sum()
        else:
            loss = loss.sum() / len(classes)

        return loss
#Maximum Margin Loss
class HMMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, gamma=1.1, ldam=False):
        super(HMMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.max_m = max_m
        self.gamma = gamma
        self.ldam = ldam

    def weight(self, freq_bias, target, args):

        index = torch.zeros_like(freq_bias, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        # plus 1 affects top-1 acc.
        cls_num_list = (index_float.sum(0).data.cpu() + 1)

        beta = args.beta

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to('cuda')

        return per_cls_weights

    def obj_margins(self, rm_obj_dists, labels, index_float, max_m):

        obj_neg_labels = 1.0 - index_float
        obj_neg_dists = rm_obj_dists * obj_neg_labels

        min_pos_prob = rm_obj_dists[:, labels.data.cpu().numpy()[0]].data
        max_neg_prob = obj_neg_dists.max(1)[0].data

        # estimate the margin between dists and gt labels
        batch_m_fg = torch.max(
            min_pos_prob - max_neg_prob,
            torch.zeros_like(min_pos_prob))[:,None]

        mask_fg = (batch_m_fg > 0).float()
        batch_fg = torch.exp(-batch_m_fg - max_m * self.gamma) * mask_fg

        batch_m_bg = torch.max(
            max_neg_prob - min_pos_prob,
            torch.zeros_like(max_neg_prob))[:,None]

        mask_ng = (batch_m_bg > 0).float()
        batch_ng = torch.exp(-batch_m_bg - max_m) * mask_ng
        batch_m = batch_ng + batch_fg
        return batch_m.data

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))

        # 1.0 - [0.5] => [0.0 ~ 0.5]
        if self.ldam :
            max_m = self.max_m - batch_m
        else:
            max_m = self.max_m

        with torch.no_grad():
            batch_hmm = self.obj_margins(x, target, index_float, max_m)

        x_m = x - batch_hmm

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output.to('cuda'), target.to('cuda'), weight=self.weight.to('cuda'))
    


class LDAMLossTrainer(Trainer):
    def __init__(self,n_per_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_per_labels = n_per_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        betas = [0, 0.99]
        beta_idx = self.state.epoch >= 2
        n_per_labels = self.n_per_labels

        effective_num = 1.0 - np.power(betas[beta_idx], n_per_labels)
        cls_weights = (1.0 - betas[beta_idx]) / np.array(effective_num)
        cls_weights = cls_weights / np.sum(cls_weights) * len(n_per_labels)
        cls_weights = torch.FloatTensor(cls_weights)
        loss_fct = LDAMLoss(cls_num_list=n_per_labels, max_m=0.5, s=30, weight=cls_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #recall = recall_score(y_true=labels, y_pred=pred)
    #precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred,average='weighted')

    return {"accuracy": accuracy, "f1": f1}
class HMMLossTrainer(Trainer):
    def __init__(self,n_per_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_per_labels = n_per_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        betas = [0, 0.99]
        beta_idx = self.state.epoch >= 2
        n_per_labels = self.n_per_labels

        effective_num = 1.0 - np.power(betas[beta_idx], n_per_labels)
        cls_weights = (1.0 - betas[beta_idx]) / np.array(effective_num)
        cls_weights = cls_weights / np.sum(cls_weights) * len(n_per_labels)
        cls_weights = torch.FloatTensor(cls_weights)
        loss_fct = HMMLoss(cls_num_list=n_per_labels, max_m=0.5, s=30, weight=cls_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    
class FocalLossTrainer(Trainer):
    def __init__(self,n_per_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_per_labels = n_per_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        betas = [0, 0.99]
        beta_idx = self.state.epoch >= 2
        n_per_labels = self.n_per_labels

        effective_num = 1.0 - np.power(betas[beta_idx], n_per_labels)
        cls_weights = (1.0 - betas[beta_idx]) / np.array(effective_num)
        cls_weights = cls_weights / np.sum(cls_weights) * len(n_per_labels)
        cls_weights = torch.FloatTensor(cls_weights)
        loss_fct =  FocalLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
       

def train(model_path, train_path, valid_path, test_path, MAXLEN, weighting):
    
    train = pd.read_csv(train_path)
    test= pd.read_csv(valid_path)
    valid = pd.read_csv(test_path)  
    train = train.sample(n=10)
    test = test.sample(n=10)
    valid = valid.sample(n=10)
    if weighting == "under":
        train = sample_under_over(train, strategy="under")
    elif weighting == "over":
        train = sample_under_over(train, strategy="over")
    else:
        train = train
        
    train_df = pd.DataFrame({
        'text':train['text'],
        'label':train['label']
    })
    valid_df = pd.DataFrame({
        'text':valid['text'],
        'label':valid['label'] 
    })
    test_df = pd.DataFrame({
        'text':test['text'],
        'label':test['label']
    })

    train_df['clean_text'] = train_df['text']
    valid_df['clean_text'] = valid_df['text']
    test_df['clean_text'] = test_df['text']
    #test['clean_text']= hero.clean(test['text'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Calculate the value counts
    value_counts =  train_df['label'].value_counts()

    # Calculate the proportions and convert to list
    proportions_list = (value_counts / len(train_df)).tolist()
    n_per_labels = value_counts.tolist()
    class_weighting = torch.tensor(proportions_list,dtype=torch.float, device=device)
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # if model_path in ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3-8B-Instruct"]:
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token = tokenizer.eos_token
    num_labels =   train_df['label'].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.config.pad_token_id = model.config.eos_token_id

    train_texts = train_df['clean_text'].values.tolist()
    valid_texts = valid_df['clean_text'].values.tolist()
    test_texts = test_df['clean_text'].values.tolist()


    max_length = MAXLEN
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)


    #train_encodings
    train_labels = train_df['label'].values.tolist()
    valid_labels = valid_df['label'].values.tolist()
    test_labels = test_df['label'].values.tolist()
    #test_partner_labels = test['label'].values.tolist()

    # convert our tokenized data into a torch Dataset
    train_dataset = DatasetPrep(train_encodings, train_labels)
    valid_dataset = DatasetPrep(valid_encodings, valid_labels)
    test_dataset = DatasetPrep(test_encodings, test_labels)
    #test_partner_dataset = DatasetPrep(test_partner_encodings, test_partner_labels)



    training_args  = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=100,
        per_device_eval_batch_size=100,
        num_train_epochs=10,
        #seed=0,
        load_best_model_at_end=True,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(valid_dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    criterion = torch.nn.CrossEntropyLoss()
    trainer = None
    if  weighting == "hmm":

        trainer =  HMMLossTrainer(
            n_per_labels = n_per_labels,
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,      # evaluation dataset
            compute_metrics=compute_metrics
        )
        trainer.train()
        result = trainer.evaluate()
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        f1 = result['eval_f1']
    elif weighting == "ldam":
        trainer = LDAMLossTrainer(
            n_per_labels = n_per_labels,
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,      # evaluation dataset
            compute_metrics=compute_metrics
        )
        trainer.train()
        result = trainer.evaluate()
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        f1 = result['eval_f1']
        
    elif weighting == "focal":
        trainer =    FocalLossTrainer(
            n_per_labels = n_per_labels,
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,      # evaluation dataset
            compute_metrics=compute_metrics
        )
        trainer.train()
        result = trainer.evaluate()
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        f1 = result['eval_f1']
    elif weighting == "custom":
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,      # evaluation dataset
            compute_metrics=compute_metrics,
        )
        trainer.train()
        result = trainer.evaluate()
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        f1 = result['eval_f1']
    elif weighting == "weighted":
        trainer =     WeightedTrainer(
            class_weighting = class_weighting,
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,      # evaluation dataset
            compute_metrics=compute_metrics
        )
        trainer.train()
        result = trainer.evaluate()
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        f1 = result['eval_f1']
    elif weighting == "under" or weighting == "over":
        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,      # evaluation dataset
            compute_metrics=compute_metrics,
        )
        trainer.train()
        result = trainer.evaluate()
        loss = result['eval_loss']
        accuracy = result['eval_accuracy']
        f1 = result['eval_f1']
            
    return loss, accuracy, f1
            


dataset = ['causal', "imdb", "agnews"]
# model_path = "google-bert/bert-base-uncased"
model_path = ["google-bert/bert-base-uncased", "roberta-base"]
MAXLEN = 100
# weighting = "weighted"


weighting_options =["hmm", "custom", "focal", "ldam", "under", "over", "weighted"]

# Main execution
dataset_paths = {
    'causal': ('/home/ptrust/experiments/data/train_causal.csv', '/home/ptrust/experiments/data/test_causal.csv', '/home/ptrust/experiments/data/test_causal.csv'),
    'imdb': ('/home/ptrust/experiments/imbalance/imdb_imbalanced.csv', '/home/ptrust/experiments/data/imdb_test.csv', '/home/ptrust/experiments/data/imdb_test.csv'),
    'agnews': ('/home/ptrust/experiments/imbalance/ag_news_imbalanced.csv', "/home/ptrust/experiments/data/train_agnews.csv", '/home/ptrust/experiments/data/train_agnews.csv')
}

model_paths = ["google-bert/bert-base-uncased", "roberta-base"]
MAXLEN = 100
weighting_options = ["hmm", "custom", "focal", "ldam", "under", "over", "weighted"]

# Initialize results list
results = []

for dataset_name, (train_path, test_path, valid_path) in dataset_paths.items():
    for model_path in model_paths:
        for weighting in weighting_options:
            loss, accuracy, f1 = train(model_path, train_path, test_path, valid_path, MAXLEN, weighting)
            results.append({
                'dataset': dataset_name,
                'model': model_path,
                'weighting': weighting,
                'loss': loss,
                'accuracy': accuracy,
                'f1': f1
            })
            print(f"Dataset: {dataset_name}, Model: {model_path}, Weighting: {weighting}")
            print(f"Loss: {loss}, Accuracy: {accuracy}, F1: {f1}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to Excel
results_df.to_excel("/home/ptrust/experiments/imbalance/experiment_results.xlsx", index=False)

    

        
