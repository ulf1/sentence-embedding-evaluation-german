[![PyPI version](https://badge.fury.io/py/sentence-embedding-evaluation-german.svg)](https://badge.fury.io/py/sentence-embedding-evaluation-german)
[![PyPi downloads](https://img.shields.io/pypi/dm/sentence-embedding-evaluation-german)](https://img.shields.io/pypi/dm/sentence-embedding-evaluation-german)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/ulf1/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/sentence-embedding-evaluation-german/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ulf1/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ulf1/sentence-embedding-evaluation-german/context:python)

# sentence-embedding-evaluation-german
Sentence embedding evaluation for German.

This library is inspired by [SentEval](https://github.com/facebookresearch/SentEval) but focuses on German language downstream tasks.


## Downstream tasks
The available downstream tasks are listed in the table below.
If you that think that a dataset is missing and should be added, please [open an issue](https://github.com/ulf1/sentence-embedding-evaluation-german/issues/new).

| task | type | text type | lang | \#train | \#test | target | info |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TOXIC | 👿 toxic comments | facebook comments | de-DE | 3244 | 944 | binary {0,1} | GermEval 2021, comments subtask 1, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| ENGAGE | 🤗 engaging comments | facebook comments | de-DE | 3244 | 944 | binary {0,1} | GermEval 2021, comments subtask 2, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| FCLAIM | ☝️ fact-claiming comments | facebook comments | de-DE | 3244 | 944 | binary {0,1} | GermEval 2021, comments subtask 3, [📁](https://github.com/germeval2021toxic/SharedTask) [📖](https://aclanthology.org/2021.germeval-1.1) |
| VMWE | ☁️ verbal idioms | newspaper | de-DE | 6652 | 1447 | binary (figuratively, literally) | GermEval 2021, verbal idioms, [📁](https://github.com/rafehr/vid-disambiguation-sharedtask) [📖](https://aclanthology.org/2020.figlang-1.29.pdf) |
| OL19-A | 👿 offensive language | tweets | de-DE | 3980 | 3031 | binary {0,1} | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) [📖](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/germeval/GermEvalSharedTask2019Iggsa.pdf) |
| OL19-B | 👿 offensive language, fine-grained | tweets | de-DE | 3980 | 3031 | 4 catg. (profanity, insult, abuse, oth.) | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) [📖](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/germeval/GermEvalSharedTask2019Iggsa.pdf) |
| OL19-C | 👿 explicit vs. implicit offense | tweets | de-DE | 1921 | 930 | binary (explicit, implicit) | GermEval 2018, [📁](https://projects.fzai.h-da.de/iggsa/data-2019/) [📖](https://corpora.linguistik.uni-erlangen.de/data/konvens/proceedings/papers/germeval/GermEvalSharedTask2019Iggsa.pdf) |
| OL18-A | 👿 offensive language | tweets | de-DE | 5009 | 3398 | binary {0,1} | GermEval 2018, [📁](https://github.com/uds-lsv/GermEval-2018-Data) |
| OL18-B | 👿 offensive language, fine-grained | tweets | de-DE | 5009 | 3398 | 4 catg. (profanity, insult, abuse, oth.) | GermEval 2018, [📁](https://github.com/uds-lsv/GermEval-2018-Data) |
| ABSD-1 | 🤷 relevance classification | 'Deutsche Bahn' customer feedback | de-DE | 19432 | 2555 | binary | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| ABSD-2 | 😃😐😡 sentiment analysis | 'Deutsche Bahn' customer feedback | de-DE | 19432 | 2555 | 3 catg. (pos., neg., neutral) | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| ABSD-3 | 🛤️ aspect categories | 'Deutsche Bahn' customer feedback | de-DE | 19432 | 2555 | 20 catg. | GermEval 2017, [📁](https://sites.google.com/view/germeval2017-absa/data) |
| MIO-S | 😃😐😡 sentiment analysis | 'Der Standard' newspaper article web comments | de-AT | 1799 | 1800 | 3 catg. | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-O | 🤷 off-topic comments | 'Der Standard' newspaper article web comments  | de-AT |  1799 | 1800  | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-I | 👿 inappropriate comments | 'Der Standard' newspaper article web comments | de-AT |   1799 | 1800  | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-D | 👿 discriminating comments| 'Der Standard' newspaper article web comments | de-AT |   1799 | 1800  | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-F | 💡 feedback comments | 'Der Standard' newspaper article web comments | de-AT |  3019  |  3019 | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-P | ✉️ personal story comments | 'Der Standard' newspaper article web comments | de-AT |  4668 | 4668 | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| MIO-A | ✴️ argumentative comments | 'Der Standard' newspaper article web comments | de-AT |  1799 | 1800 | binary | One Million Posts Corpus, [📁](https://github.com/OFAI/million-post-corpus/releases/tag/v1.0.0) |
| SBCH-S | 😃😐😡 sentiment analysis | 'chatmania' app comments, only comments labelled as Swiss German are included | gsw | 394 | 394  | 3 catg. | SB-CH Corpus, [📁](https://github.com/spinningbytes/SB-CH) |
| SBCH-L | ⛰️ dialect classification | 'chatmania' app comments | gsw | 748 | 748 | binary | SB-CH Corpus, [📁](https://github.com/spinningbytes/SB-CH) |
| ARCHI | ⛰️ dialect classification | Audio transcriptions of interviews in four dialect regions of Switzerland | gsw | 18809 | 4743 | 4 catg. | ArchiMob, [📁](https://www.spur.uzh.ch/en/departments/research/textgroup/ArchiMob.html) [📖](https://aclanthology.org/L16-1641) |
| LSDC | 🌊 dialect classification | several genres (e.g. formal texts, fairytales, novels, poetry, theatre plays) from the 19th to 21st centuries. Extincted Lower Prussia excluded. Gronings excluded due to lack of test examples. | nds | 74140 | 8602 | 14 catg. | Lower Saxon Dialect Classification, [📁](https://github.com/Helsinki-NLP/LSDC) [📖](https://www.aclweb.org/anthology/2020.vardial-1.3) |
| KLEX-P | 🤔 text level | Conceptual complexity classification of texts written for adults (Wikipedia), children between 6-12 (Klexikon), and beginner readers (MiniKlexikon); Paragraph split indicated by `<eop>` or ` * ` | de | 8264 | 8153 | 3 catg. | [📁](https://zenodo.org/record/6319803) [📖](https://aclanthology.org/2021.konvens-1.23) |


## Download datasets

```sh
bash download-datasets.sh
```

Check if files were actually downloaded
```sh
find ./datasets/**/ -exec ls -lh {} \;
```

## Usage example
Import the required Python packages.

```py
from typing import List
import sentence_embedding_evaluation_german as seeg
import torch
```

### Step (1) Load your pretrained model
In the following example, we generate a random embedding matrix for demonstration purposes.
```py
# (1) Instantiate an embedding model
emb_dim = 512
vocab_sz = 128
emb = torch.randn((vocab_sz, emb_dim), requires_grad=False)
emb = torch.nn.Embedding.from_pretrained(emb)
assert emb.weight.requires_grad == False
```

### Step (2) Specify your `preprocessor` function
You need to specify your own preprocessing routine.
The `preprocessor` function must convert a list of strings `batch` (`List[str]`)
into a list of feature vectors, or resp. a list of sentence embeddings (`List[List[float]]`).
In the following example, we generate some sort of token IDs, retrieve the vectors from our random matrix, and average these to feature vectors for demonstration purposes.
```py
# (2) Specify the preprocessing
def preprocesser(batch: List[str], params: dict=None) -> List[List[float]]:
    """ Specify your embedding or pretrained encoder here
    Paramters:
    ----------
    batch : List[str]
        A list of sentence as string
    params : dict
        The params dictionary
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
```

### Step (3) Training settings
We suggest to train a final layer with bias term (`'bias':True`),
on a loss function weighted by the class frequency (`'balanced':True`),
a batch size of 128, an over 500 epochs without early stopping.
```py
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
```

### Step (4) Downstream tasks
We suggest to run the following downstream tasks.
`FCLAIM` flags comments that requires manual fact-checking because these contain reasoning, arguments or claims that might be false.
`VMWE` differentiates texts with figurative or literal multi-word expressions.
`OL19-C` distincts between explicit and implicit offensive language.
`ABSD-2` is a sentiment analysis dataset with customer reviews.
These four dataset so far can be assumed to be Standard German from Germany (de-DE).
`MIO-P` flags Austrian German (de-AT) comments if these contain personal stories.
`ARCHI` is a Swiss (gsw), and `LSDC` a Lower German (nds) dialect identification task.

```py
# (4) Specify downstream tasks
downstream_tasks = ['FCLAIM', 'VMWE', 'OL19-C', 'ABSD-2', 'MIO-P', 'ARCHI', 'LSDC']
```

### Step (5) Run the experiments
Finally, start the evaluation. 
The suggested downstream tasks (step 4) with 500 epochs (step 3) 
might requires 10-40 minutes but it's highly dependent on your computing resources.
So grab a ☕ or 🍵.
```py
# (5) Run experiments
results = seeg.evaluate(downstream_tasks, preprocesser, **params)
```

## Demo notebooks
Start Jupyter
```sh
source .venv/bin/activate
jupyter lab
```

Open an demo notebook

- [Generic demo](demo/Jupyter%20Demo.ipynb)
- [deepset example](demo/deepset%20example.ipynb)
- [fasttext example](demo/fastText%20example.ipynb)
- [SBert example](demo/SBert%20example.ipynb)


## Appendix

### Installation & Downloads
The `sentence-embedding-evaluation-german` [git repo](http://github.com/ulf1/sentence-embedding-evaluation-german) is available as [PyPi package](https://pypi.org/project/sentence-embedding-evaluation-german)

```sh
pip install sentence-embedding-evaluation-german
pip install git+ssh://git@github.com/ulf1/sentence-embedding-evaluation-german.git
```

You need to download the datasets as well.
If you run the following code, the datasets should be in a folder `./datasets`.

```sh
wget -q "https://raw.githubusercontent.com/ulf1/sentence-embedding-evaluation-german/main/download-datasets.sh" -O download-datasets.sh 
bash download-datasets.sh
```


### Development work for this package

#### Install a virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
pip install -r requirements-demo.txt --no-cache-dir
```

(If your git repo is stored in a folder with whitespaces, then don't use the subfolder `.venv`. Use an absolute path without whitespaces.)

#### Install conda environment for GPU

```sh
conda install -y pip
conda create -y --name gpu-venv-seeg python=3.9 pip
conda activate gpu-venv-seeg
# install CUDA support
conda install -y cudatoolkit=11.3.1 cudnn=8.3.2 -c conda-forge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install torch==1.12.1+cu113 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
# install other packages
pip install -r requirements.txt --no-cache-dir
pip install -r requirements-dev.txt --no-cache-dir
pip install -r requirements-demo.txt --no-cache-dir
watch -n 0.5 nvidia-smi
```

#### Python commands

* Jupyter for the examples: `jupyter lab`
* Check syntax: `flake8 --ignore=F401 --exclude=$(grep -v '^#' .gitignore | xargs | sed -e 's/ /,/g')`

#### Publish package

```sh
pandoc README.md --from markdown --to rst -s -o README.rst
python setup.py sdist 
twine upload -r pypi dist/*
```

#### Clean up 

```sh
find . -type f -name "*.pyc" | xargs rm
find . -type d -name "__pycache__" | xargs rm -r
rm -r .pytest_cache
rm -r .venv
```

### New Dataset recommendation
If you want to recommend another or a new dataset, please [open an issue](https://github.com/ulf1/sentence-embedding-evaluation-german/issues/new).


### Troubleshooting
If you have troubles to get this package running, please [open an issue](https://github.com/ulf1/sentence-embedding-evaluation-german/issues/new) for support.


### Contributing
Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add commits, and [open a pull request](https://github.com/ulf1/sentence-embedding-evaluation-german/compare/).


### Citation
If you want to use this package in a research paper, please [open an issue](https://github.com/ulf1/sentence-embedding-evaluation-german/issues/new) because we have not yet decided how to make this package citable. You should at least mention the PyPi version in your paper to ensure reproducibility.

You certainly need to cite the actual evaluation datasets in your paper. Please check the hyperlinks in the info column of the [table above](#downstream-tasks).
