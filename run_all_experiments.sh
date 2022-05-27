#!/usr/bin/bash

set -x

# English
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --data_split dev --language en --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --data_split dev --language en --ud_system trankit -r final_trankit

python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --data_split test --language en --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --data_split test --language en --ud_system trankit -r final_trankit

python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --data_split eval --language en --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/en/gold --data_split eval --language en --ud_system trankit -r final_trankit

# Dutch
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --data_split dev --language nl --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --data_split dev --language nl --ud_system trankit -r final_trankit

python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --data_split test --language nl --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/nl/gold --data_split test --language nl --ud_system trankit -r final_trankit

# German
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --data_split dev --language de --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --data_split dev --language de --ud_system trankit -r final_trankit

python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --data_split test --language de --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/de/gold --data_split test --language de --ud_system trankit -r final_trankit

# Italian
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --data_split dev --language it --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --data_split dev --language it --ud_system trankit -r final_trankit

python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --data_split test --language it --ud_system stanza -r final_stanza
python pmb_inference.py -p ../../data/pmb_dataset/pmb-extracted/pmb-4.0.0/data/it/gold --data_split test --language it --ud_system trankit -r final_trankit
