#!/usr/bin/env bash

TOPICS=$1
START_MONTH=$2
END_MONTH=$3

for i in $(seq $START_MONTH $END_MONTH); do
  mkdir -p data/results/topics-$TOPICS/$i
  ./socialization $TOPICS data/$i/src_blobs.txt data/$i/tgt_blobs.txt data/$i/edges.txt data/$i/subreddits.txt data/results/topics-$TOPICS/$i
done

apptainer exec --nv --bind /gscratch  covid.sif python covid/covid.py $END_MONTHS $TOPICS
