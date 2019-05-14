# !/bin/bash

cd ../
LUCENE_BENCH_HOME=`pwd`

echo export LUCENE_BENCH_HOME=$LUCENE_BENCH_HOME >> ~/.bashrc

source ~/.bashrc

cd $LUCENE_BENCH_HOME/util

python src/python/setup.py

#echo export LUCENE_BENCH_HOME= >> ~/.bashrc
