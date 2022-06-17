#!/usr/bin/bash

set -x

# English gold only
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --input_file data/seq2seq/seq2seq_gold_only_dev.txt --data_split dev --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --input_file data/seq2seq/seq2seq_gold_only_test.txt --data_split test --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --input_file data/seq2seq/seq2seq_gold_only_eval.txt --data_split eval --results_file results_strict_indices_gold.csv

# English gold + silver
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --input_file data/seq2seq/seq2seq_silver_and_gold_dev.txt --data_split dev --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --input_file data/seq2seq/seq2seq_silver_and_gold_test.txt --data_split test --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --input_file data/seq2seq/seq2seq_silver_and_gold_eval.txt --data_split eval --results_file results_strict_indices_gold_silver.csv
