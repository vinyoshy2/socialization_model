#!/usr/bin/env bash

TOPICS=$1
START_MONTH=$2
END_MONTH=$3

for i in $(seq $START_MONTH $END_MONTH); do
  mkdir -p data/results/K=$TOPICS/$i
  ./socialization-opt data/1 data/results/K=$TOPICS/$i --topics $TOPICS --iters 1000 --warmup 200 --alpha-vocab 1 --alpha-topics 1 --alpha-edges 1
done

# apptainer exec --nv --bind /gscratch  covid.sif python covid/covid.py $END_MONTHS $TOPICS
