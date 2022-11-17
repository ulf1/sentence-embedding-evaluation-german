
import sys
sys.path.append('..')

from typing import List
import sentence_embedding_evaluation_german as seeg
import torch

import sentence_transformers as sbert
model_sbert = sbert.SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# (1) Instantiate your Embedding model
model_sbert = sbert.SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2')

# (2) Specify the preprocessing
def preprocesser(batch: List[str], params: dict=None) -> List[List[float]]:
    """ Specify your embedding or pretrained encoder here
    Paramters:
    ----------
    params : dict
        The params dictionary
    batch : List[str]
        A list of sentence as string
    Returns:
    --------
    List[List[float]]
        A list of embedding vectors
    """
    features = model_sbert.encode(batch)
    return features

# (3) Training settings
params = {
    'datafolder': '../datasets',
    'bias': True,
    'balanced': True,
    'batch_size': 128,
    'num_epochs': 250,
    # 'early_stopping': True,
    # 'split_ratio': 0.1,  # if early_stopping=True
    # 'patience': 10,  # if early_stopping=True
}

# (4) Specify downstream tasks
downstream_tasks = ['FCLAIM', 'VMWE', 'OL19-C', 'ABSD-2', 'MIO-P', 'ARCHI', 'LSDC']

# (5) Run experiments
results = seeg.evaluate(downstream_tasks, preprocesser, **params)


import json
with open("seeg-results-sbert.json", 'w') as fp:
    json.dump(results, fp, indent=2)


print("Task | Epochs | N train | N test")
for res in results:
    print(f"{res['task']:>7s}: {res['epochs']:5d} {res['train']['num']:6d} {res['test']['num']:6d}")


metric = 'f1-balanced'  # 'f1', 'f1-balanced'
print(metric)
print('  Task | train | test')
for res in results:
    print(f"{res['task']:>7s}: {res['train'][metric]:6.3f} {res['test'][metric]:6.3f}")
