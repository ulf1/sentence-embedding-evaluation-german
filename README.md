[![PyPI version](https://badge.fury.io/py/sentence-embedding-evaluation-german.svg)](https://badge.fury.io/py/sentence-embedding-evaluation-german)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/linguistik/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/linguistik/sentence-embedding-evaluation-german/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/linguistik/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/linguistik/sentence-embedding-evaluation-german/context:python)

# sentence-embedding-evaluation-german
Sentence embedding evaluation for German.

This library is inspired by [SentEval](https://github.com/facebookresearch/SentEval) but focuses on German language downstream tasks.


## Downstream tasks

| task | type | properties | \#train | \#test | target | info |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TOXIC | 👿 toxic comments | facebook comments | 2596 | 944 | binary {0,1} | GermEval 2021, comments subtask 1, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| ENGAGE | 🤗 engaging comments | facebook comments | 2596 | 944 | binary {0,1} | GermEval 2021, comments subtask 2, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| FCLAIM | ☝️ fact-claiming comments | facebook comments | 2596 | 944 | binary {0,1} | GermEval 2021, comments subtask 3, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| VMWE | verbal idioms | newspaper | 6651 | 1446 | binary (figuratively, literally) | GermEval 2021, verbal idioms, [📁](https://github.com/rafehr/vid-disambiguation-sharedtask) [📖](https://aclanthology.org/2020.figlang-1.29.pdf) |
| OL19A | 👿 offensive language | tweets | 3184 | 3030 | binary {0,1} | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) |
| OL19B | 👿 offensive language, fine-grained | tweets | 3184 | 3030 | 4 catg. (profanity, insult, abuse, oth.) | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) |
| OL19C | 👿 explicit vs. implicit offense | tweets | 1920 | 929 | binary (explicit, implicit) | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) |
| OL18A | 👿 offensive language | tweets | 4007 | 3397 | binary {0,1} | GermEval 2018, [📁](https://github.com/uds-lsv/GermEval-2018-Data) |
| OL18B | 👿 offensive language, fine-grained | tweets | 4007 | 3397 | 4 catg. (profanity, insult, abuse, oth.) | GermEval 2018, [📁](https://github.com/uds-lsv/GermEval-2018-Data) |
| ABSD1 | 🛤️ relevance classification | Deutsche Bahn customer feedback | 19431 | 2565 | binary | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| ABSD2 | 🛤️ sentiment analysis | Deutsche Bahn customer feedback | 19431 | 2565 | 3 catg. (pos., neg., neutral) | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| ABSD3 | 🛤️ aspect categories | Deutsche Bahn customer feedback | 19431 | 2565 | 20 catg. | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |


## Download datasets

```sh
bash download-datasets.sh
```

## Usage example

```py
from typing import List
import sentence_embedding_evaluation_german as seeg
import torch

# (1) Instantiate your Embedding model
emb_dim = 512
vocab_sz = 128
emb = torch.randn((vocab_sz, emb_dim), requires_grad=False)
emb = torch.nn.Embedding.from_pretrained(emb)
assert emb.weight.requires_grad == False

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
    features = []
    for sent in batch:
        try:
            ids = torch.tensor([ord(c) % 128 for c in sent])
        except:
            print(sent)
        h = emb(ids)
        features.append(h.mean(axis=0))
    features = torch.stack(features, dim=0)
    return features

# (3) Training settings
params = {
    'datafolder': '../datasets',
    'batch_size': 128, 
    'num_epochs': 20,
    # 'early_stopping': True,
    # 'split_ratio': 0.2,  # if early_stopping=True
    # 'patience': 5,  # if early_stopping=True
}

# (4) Specify downstream tasks
downstream_tasks = [
    'TOXIC', 'ENGAGE', 'FCLAIM', 'VMWE',
    'OL19A', 'OL19B', 'OL19C',
    'ABSD1', 'ABSD2', 'ABSD3']

# (5) Run experiments
results = seeg.evaluate(downstream_tasks, preprocesser, **params)
```

## Appendix

### Installation
The `sentence-embedding-evaluation-german` [git repo](http://github.com/linguistik/sentence-embedding-evaluation-german) is available as [PyPi package](https://pypi.org/project/sentence-embedding-evaluation-german)

```sh
pip install sentence-embedding-evaluation-german
pip install git+ssh://git@github.com/linguistik/sentence-embedding-evaluation-german.git
```

### Install a virtual environment

```sh
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
pip install -r requirements-demo.txt --no-cache-dir
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`

Publish

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

### Clean up 

```sh
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```


### Support
Please [open an issue](https://github.com/linguistik/sentence-embedding-evaluation-german/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/linguistik/sentence-embedding-evaluation-german/compare/).
