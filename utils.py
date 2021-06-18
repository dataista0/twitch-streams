from IPython.core.display import display, HTML

import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import subprocess
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import pathlib
BASE_PATH = pathlib.Path(__file__).parent.absolute()

BASE_MODELS_PATH = BASE_PATH/"data/models"


def get_trainer(model, ds_train, ds_eval, args=None):
    if args is None:
        args = get_train_args()
    return Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_eval,
                   compute_metrics=compute_metrics)


def tokenize(tokenizer, df_train, df_val, df_test):
    train_tokenized = tokenizer(df_train.text.tolist(), padding="max_length", truncation=True)
    val_tokenized = tokenizer(df_val.text.tolist(), padding="max_length", truncation=True)
    test_tokenized = tokenizer(df_test.text.tolist(), padding="max_length", truncation=True)
    train_tokenized['label'] = df_train.label.tolist()
    val_tokenized['label'] = df_val.label.tolist()
    ds_train = [dict(zip(train_tokenized,t)) for t in zip(*train_tokenized.values())]
    ds_val = [dict(zip(val_tokenized,t)) for t in zip(*val_tokenized.values())]
    ds_test = [dict(zip(test_tokenized,t)) for t in zip(*test_tokenized.values())]
    return ds_train, ds_val, ds_test


def load_dfs():
    df_train = pd.read_csv('data/nlp-getting-started/train.csv')
    df_train = df_train[['text', 'target']].rename(columns={'target': 'label'})
    df_test = pd.read_csv('data/nlp-getting-started/test.csv')
    
    df_test = df_test[['id', 'text']]
    return df_train, df_test


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def get_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model




def nb_set_width(width='100'):
    """ Increase (or decrease) the width of a jupyter notebook to fit the specified pct of the screen """
    display(HTML("<style>.container { width:" + width + "% !important; }</style>"))


def get_train_args(model_id="default", batch_size=8, epochs=1, **kwargs):
    path = BASE_MODELS_PATH / model_id
    return TrainingArguments(path, 
                             per_device_train_batch_size=batch_size,
                             per_device_eval_batch_size=batch_size,
                             num_train_epochs=epochs,
                             evaluation_strategy="epoch", **kwargs)
    

def get_submission(trainer, ds_test, file_name='default_pred.csv'):
    df_res = pd.read_csv('data/nlp-getting-started/sample_submission.csv')
    preds = trainer.predict(ds_test)
    final_preds = F.softmax(torch.from_numpy(preds.predictions), dim=-1)
    final_binary_preds = (final_preds[:, 1] > 0.5).numpy().astype(int)
    df_res['target'] = final_binary_preds
    df_res.to_csv(file_name, index=False)
    return df_res


def submit(file_name='default_pred.csv', message="A submission"):
    session = subprocess.Popen(['kaggle', 'competitions', 'submit', '-f', file_name, '-m', f'"{message}"', 'nlp-getting-started'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = session.communicate()
    print("stdout=")
    print(stdout.decode("utf-8"))
    print("stderr=")
    print(stderr.decode("utf-8"))