#!/bin/bash

FOLDER=$(pwd)

mkdir -p datasets/germeval21
wget -q "https://raw.githubusercontent.com/germeval2021toxic/SharedTask/main/Data%20Sets/GermEval21_TrainData.csv" -O datasets/germeval21/train.csv
wget -q "https://raw.githubusercontent.com/germeval2021toxic/SharedTask/main/Data%20Sets/GermEval21_TestData.csv" -O datasets/germeval21/test.csv

mkdir -p datasets/germeval21vmwe
wget -q "https://raw.githubusercontent.com/rafehr/vid-disambiguation-sharedtask/main/data/train/train.tsv" -O datasets/germeval21vmwe/train.tsv
wget -q "https://raw.githubusercontent.com/rafehr/vid-disambiguation-sharedtask/main/data/test/test.tsv" -O datasets/germeval21vmwe/test.tsv

mkdir -p datasets/germeval19
wget -q "https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/09/germeval2019.training_subtask1_2_korrigiert.txt" -O datasets/germeval19/train12.txt
wget -q "https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/05/germeval2019.training_subtask3.txt" -O datasets/germeval19/train3.txt
wget -q "https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/08/germeval2019GoldLabelsSubtask1_2.txt" -O datasets/germeval19/gold12.txt
wget -q "https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/08/germeval2019GoldLabelsSubtask3.txt" -O datasets/germeval19/gold3.txt

mkdir -p datasets/germeval18
wget -q "https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/05/germeval2018.training.txt" -O datasets/germeval18/train.txt
wget -q "https://projects.fzai.h-da.de/iggsa/wp-content/uploads/2019/05/germeval2018.test_.txt" -O datasets/germeval18/test.txt

mkdir -p datasets/germeval17
wget -q "http://ltdata1.informatik.uni-hamburg.de/germeval2017/train-2017-09-15.tsv" -O datasets/germeval17/train.tsv
wget -q "http://ltdata1.informatik.uni-hamburg.de/germeval2017/test_syn-2017-09-15.tsv" -O datasets/germeval17/test.tsv
# wget -q "http://ltdata1.informatik.uni-hamburg.de/germeval2017/test_dia-2017-09-15.tsv" -O datasets/germeval17/test-dia.tsv

mkdir -p datasets/1mio
wget -q "https://github.com/OFAI/million-post-corpus/releases/download/v1.0.0/million_post_corpus.tar.bz2" -O datasets/1mio/million_post_corpus.tar.bz2
cd datasets/1mio && tar -xvf million_post_corpus.tar.bz2
cd "$FOLDER"
mv datasets/1mio/million_post_corpus/corpus.sqlite3 datasets/1mio/corpus.sqlite3
rm -r datasets/1mio/million_post_corpus
rm datasets/1mio/million_post_corpus.tar.bz2

mkdir -p datasets/sbch
wget -q "https://raw.githubusercontent.com/spinningbytes/SB-CH/master/sentiment.csv" -O datasets/sbch/sentiment.csv
wget -q "https://raw.githubusercontent.com/spinningbytes/SB-CH/master/chatmania.csv" -O datasets/sbch/chatmania.csv
