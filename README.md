# thesis
Source for my master's thesis system for Information Science.

## Installation
1. (Optional) create and activate a virtual environment (tested with python 3.8+)
2. (Optional) for visualization install `graphviz`. 
   1. On Debian-based systems: `apt install graphviz libgraphviz-dev pkg-config`
   2. The `networkx` extras: `pip install networkx[default,extras]`
3. Install dependencies with `pip install -r requirements.txt`
4. Run `fix_all.sh` to format and test the project

## Data
The data used comes from the Parallel Meaning Bank project (https://pmb.let.rug.nl/).
- This project uses the version 4.0.0 of the PMB dataset, which can be downloaded from here: https://pmb.let.rug.nl/data.php

## TODO
- [x] Maybe use roles spec explicitly in parsing sbn?
  * related question: is it possible that a constant looks like an index and *is a valid index* (in the range of possible indices for the current file)? Example with invalid index: `pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn` Maybe indicated by the role since such ambiguity should not be possible?
- [x] Try out https://github.com/nlp-uoregon/trankit (supposedly better performance than stanza in some areas: https://trankit.readthedocs.io/en/latest/performance.html#universal-dependencies-v2-5)
- [x] Read a UD parse into nx.Graph (from file (connl?)? in a program directly using sentence?)
- [ ] Support enhanced UD annotations (need CoreNLP binding: https://stanfordnlp.github.io/CoreNLP/depparse.html or keep an eye on this: https://github.com/stanfordnlp/stanza/issues/359) these are essential for case markings
- [ ] convert UD pos to start of wordnet sense (`<lemma>.<sn_style_pos>`) sense number is for later / different component
- [ ] attach box nodes in `I` UDGraph (at least starting box), figure out new box indicators in UD

## Examples to try
* **p00/d0004**: `entity` that combines multiple subtypes
* **p00/d0801**: multiple boxes
* **p00/d1593**: negation
* **p00/d2719**: case marking / pivot -> extended UD needed!
* **p03/d2003**: named entity 'components' combining in single item
* **p04/d0778**: double negation
* **p04/d1646**: 'basic' sentence to start with

## Plan
* remove based on incoming edge deprel
* create new intermediate graph (or node/edge) type that has a meta data about decision (= labels with both UD and SBN information)

