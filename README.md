[![PyPI version](https://badge.fury.io/py/sentence-embedding-evaluation-german.svg)](https://badge.fury.io/py/sentence-embedding-evaluation-german)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ulf1/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/sentence-embedding-evaluation-german/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ulf1/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/sentence-embedding-evaluation-german/context:python)

# sentence-embedding-evaluation-german
Sentence embedding evaluation for German.

This library is inspired by [SentEval](https://github.com/facebookresearch/SentEval) but focuses on German language downstream tasks.


## Downstream tasks

| task | type | properties | \#train | \#test | target | info |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TOXIC | 👿 toxic comments | facebook comments | 3244 | 944 | binary {0,1} | GermEval 2021, comments subtask 1, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| ENGAGE | 🤗 engaging comments | facebook comments | 3244 | 944 | binary {0,1} | GermEval 2021, comments subtask 2, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| FCLAIM | ☝️ fact-claiming comments | facebook comments | 3244 | 944 | binary {0,1} | GermEval 2021, comments subtask 3, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| VMWE | verbal idioms | newspaper | 6652 | 1447 | binary (figuratively, literally) | GermEval 2021, verbal idioms, [📁](https://github.com/rafehr/vid-disambiguation-sharedtask) [📖](https://aclanthology.org/2020.figlang-1.29.pdf) |
| OL19-A | 👿 offensive language | tweets | 3980 | 3031 | binary {0,1} | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) [📖](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/germeval/GermEvalSharedTask2019Iggsa.pdf) |
| OL19-B | 👿 offensive language, fine-grained | tweets | 3980 | 3031 | 4 catg. (profanity, insult, abuse, oth.) | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) [📖](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/germeval/GermEvalSharedTask2019Iggsa.pdf) |
| OL19-C | 👿 explicit vs. implicit offense | tweets | 1921 | 930 | binary (explicit, implicit) | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) [📖](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/germeval/GermEvalSharedTask2019Iggsa.pdf) |
| OL18-A | 👿 offensive language | tweets | 5009 | 3398 | binary {0,1} | GermEval 2018, [📁](https://github.com/uds-lsv/GermEval-2018-Data) |
| OL18-B | 👿 offensive language, fine-grained | tweets | 5009 | 3398 | 4 catg. (profanity, insult, abuse, oth.) | GermEval 2018, [📁](https://github.com/uds-lsv/GermEval-2018-Data) |
| ABSD-1 | 🛤️ relevance classification | 'Deutsche Bahn' customer feedback, `lang:de-DE` | 19432 | 2555 | binary | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| ABSD-2 | 🛤️ Sentiment analysis | 'Deutsche Bahn' customer feedback, `lang:de-DE` | 19432 | 2555 | 3 catg. (pos., neg., neutral) | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| ABSD-3 | 🛤️ aspect categories | 'Deutsche Bahn' customer feedback, `lang:de-DE` | 19432 | 2555 | 20 catg. | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| MIO-S | Sentiment analysis | 'Der Standard' newspaper article web comments, `lang:de-AT` | 1799 | 1800 | 3 catg. | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-O | off-topic comments | 'Der Standard' newspaper article web comments, `lang:de-AT` |  1799 | 1800  | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-I | inappropriate comments | 'Der Standard' newspaper article web comments, `lang:de-AT` |  1799 | 1800  | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-D | discriminating comments| 'Der Standard' newspaper article web comments, `lang:de-AT` |  1799 | 1800  | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-F | feedback comments | 'Der Standard' newspaper article web comments, `lang:de-AT` | 3019  |  3019 | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-P | personal story comments | 'Der Standard' newspaper article web comments, `lang:de-AT` | 4668 | 4668 | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-A | argumentative comments | 'Der Standard' newspaper article web comments, `lang:de-AT` | 1799 | 1800 | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| SBCH-L | Swiss German detection | 'chatmania' app comments, `lang:gsw` | 748 | 748 | binary | SB-CH Corpus, [📁](https://github.com/spinningbytes/SB-CH) |
| SBCH-S | Sentiment analysis | 'chatmania' app comments, only comments labelled as Swiss German are included, `lang:gsw` |  394 | 394  | 3 catg. | SB-CH Corpus, [📁](https://github.com/spinningbytes/SB-CH) |
| ARCHI | Swiss German Dialect Classification | `lang:gsw` | 18809 | 4743 | 4 catg. | ArchiMob, [📁](https://www.spur.uzh.ch/en/departments/research/textgroup/ArchiMob.html) [📖](https://aclanthology.org/W19-1401/) |
| LSDC | Lower Saxon Dialect Classification | `lang:nds` | 74140 | 8602 | 14 catg. | LSDC, [📁](https://github.com/Helsinki-NLP/LSDC) [📖](https://www.aclweb.org/anthology/2020.vardial-1.3) |




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
    'datafolder': './datasets',
    'bias': True,
    'balanced': True,
    'batch_size': 128, 
    'num_epochs': 500,
    # 'early_stopping': True,
    # 'split_ratio': 0.2,  # if early_stopping=True
    # 'patience': 5,  # if early_stopping=True
}

# (4) Specify downstream tasks
downstream_tasks = ['FCLAIM', 'VMWE', 'OL19-C', 'ABSD-2', 'MIO-P', 'ARCHI', 'LSDC']

# (5) Run experiments
results = seeg.evaluate(downstream_tasks, preprocesser, **params)
```

## Appendix

### Installation
The `sentence-embedding-evaluation-german` [git repo](http://github.com/ulf1/sentence-embedding-evaluation-german) is available as [PyPi package](https://pypi.org/project/sentence-embedding-evaluation-german)

```sh
pip install sentence-embedding-evaluation-german
pip install git+ssh://git@github.com/ulf1/sentence-embedding-evaluation-german.git
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
Please [open an issue](https://github.com/ulf1/sentence-embedding-evaluation-german/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/ulf1/sentence-embedding-evaluation-german/compare/).
