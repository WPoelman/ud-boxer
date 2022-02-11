# thesis
Source for my master's thesis system for Information Science.

## Installation
1. (Optional) create and activate a virtual environment (tested with python 3.8+)
2. (Optional) for visualization install `graphviz`. 
   1. On Debian-based systems: `apt install graphviz libgraphviz-dev pkg-config`
   2. The `networkx` extras: `pip install networkx[default,extras]`
3. Install dependencies with `pip install -r requirements.txt`
4. Run `pytest` to go through all tests

## TODO
- [ ] Maybe use roles spec explicitly in parsing sbn?
  * related question: is it possible that a constant looks like an index and *is a valid index* (in the range of possible indices for the current file)? Example with invalid index: `pmb-4.0.0/data/en/silver/p15/d3131/en.drs.sbn` Maybe indicated by the role since such ambiguity should not be possible?
- [ ] Try out https://github.com/nlp-uoregon/trankit (supposedly better performance than stanza in some areas: https://trankit.readthedocs.io/en/latest/performance.html#universal-dependencies-v2-5)
- [ ] Read a UD parse into nx.Graph (from file (connl?)? in a program directly using sentence?)

## SBN output error?
Most ids connect parts together with either a '-' or '_', since there are only three in the entire EN dataset with spaces in them, this seems like an error.

* `pmb-4.0.0/data/en/silver/p92/d0079/en.drs.sbn`: `cocktails at.v.01`
* `pmb-4.0.0/data/en/silver/p80/d0541/en.drs.sbn`: `of developed.v.01 `
* `pmb-4.0.0/data/en/bronze/p60/d0288/en.drs.sbn`: `2 million-dollar.a.01`

- In it/de/nl there are more sense ids with spaces in them, but again, not a lot:
   * IT: 3 / 106737
   * DE: 22 / 160692
   * NL: 23 / 31172
