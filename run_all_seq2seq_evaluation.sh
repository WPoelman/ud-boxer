#!/usr/bin/bash

set -x

# English gold only
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --language en --input_file data/seq2seq/en/gold_dev.txt --data_split dev --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --language en --input_file data/seq2seq/en/gold_test.txt --data_split test --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --language en --input_file data/seq2seq/en/gold_eval.txt --data_split eval --results_file results_strict_indices_gold.csv

# # English gold + silver
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --language en --input_file data/seq2seq/en/gold_silver_dev.txt --data_split dev --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --language en --input_file data/seq2seq/en/gold_silver_test.txt --data_split test --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --language en --input_file data/seq2seq/en/gold_silver_eval.txt --data_split eval --results_file results_strict_indices_gold_silver.csv

# Dutch
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --language nl --input_file data/seq2seq/nl/gold_dev.txt --data_split dev --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --language nl --input_file data/seq2seq/nl/gold_test.txt --data_split test --results_file results_strict_indices_gold.csv

python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --language nl --input_file data/seq2seq/nl/gold_silver_dev.txt --data_split dev --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --language nl --input_file data/seq2seq/nl/gold_silver_test.txt --data_split test --results_file results_strict_indices_gold_silver.csv

python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --language nl --input_file data/seq2seq/nl/gold_silver_bronze_dev.txt --data_split dev --results_file results_strict_indices_gold_silver_bronze.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --language nl --input_file data/seq2seq/nl/gold_silver_bronze_test.txt --data_split test --results_file results_strict_indices_gold_silver_bronze.csv

# German
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --language de --input_file data/seq2seq/de/gold_dev.txt --data_split dev --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --language de --input_file data/seq2seq/de/gold_test.txt --data_split test --results_file results_strict_indices_gold.csv

python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --language de --input_file data/seq2seq/de/gold_silver_dev.txt --data_split dev --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --language de --input_file data/seq2seq/de/gold_silver_test.txt --data_split test --results_file results_strict_indices_gold_silver.csv

python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --language de --input_file data/seq2seq/de/gold_silver_bronze_dev.txt --data_split dev --results_file results_strict_indices_gold_silver_bronze.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --language de --input_file data/seq2seq/de/gold_silver_bronze_test.txt --data_split test --results_file results_strict_indices_gold_silver_bronze.csv

# Italian
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --language it --input_file data/seq2seq/it/gold_dev.txt --data_split dev --results_file results_strict_indices_gold.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --language it --input_file data/seq2seq/it/gold_test.txt --data_split test --results_file results_strict_indices_gold.csv

python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --language it --input_file data/seq2seq/it/gold_silver_dev.txt --data_split dev --results_file results_strict_indices_gold_silver.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --language it --input_file data/seq2seq/it/gold_silver_test.txt --data_split test --results_file results_strict_indices_gold_silver.csv

python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --language it --input_file data/seq2seq/it/gold_silver_bronze_dev.txt --data_split dev --results_file results_strict_indices_gold_silver_bronze.csv
python seq2seq_eval.py --starting_path ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --language it --input_file data/seq2seq/it/gold_silver_bronze_test.txt --data_split test --results_file results_strict_indices_gold_silver_bronze.csv
