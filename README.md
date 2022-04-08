[![PyPI version](https://badge.fury.io/py/sentence-embedding-evaluation-german.svg)](https://badge.fury.io/py/sentence-embedding-evaluation-german)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/linguistik/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/linguistik/sentence-embedding-evaluation-german/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/linguistik/sentence-embedding-evaluation-german.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/linguistik/sentence-embedding-evaluation-german/context:python)

# sentence-embedding-evaluation-german
This library is inspired by [SentEval](https://github.com/facebookresearch/SentEval) but focuses on German language downstream tasks.


## Usage

Table of Contents

* [Use Case 1](#use-case-1)


### Use Case 1


## Appendix

### Installation
The `sentence-embedding-evaluation-german` [git repo](http://github.com/linguistik/sentence-embedding-evaluation-german) is available as [PyPi package](https://pypi.org/project/sentence-embedding-evaluation-german)

```sh
pip install sentence-embedding-evaluation-german
pip install git+ssh://git@github.com/linguistik/sentence-embedding-evaluation-german.git
```

### Install a virtual environment

```sh
python3.6 -m venv .venv
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
* Run Unit Tests: `PYTHONPATH=. pytest`

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
