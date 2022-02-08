# thesis
Source for my master's thesis system for Information Science.

## Installation
1. (Optional) create and activate a virtual environment (tested with python 3.8+)
2. (Optional) for visualization install `graphviz`. 
   1. On Debian-based systems: `apt install graphviz libgraphviz-dev pkg-config`
   2. The `networkx` extras: `pip install networkx[default,extras]`
3. Install dependencies with `pip install -r requirements.txt`


## SBN output error?
Most ids connect parts together with either a '-' or '_', since there are only three in the entire EN dataset with spaces in them, this seems like an error.

* `pmb-4.0.0/data/en/silver/p92/d0079/en.drs.sbn`: `cocktails at.v.01`
* `pmb-4.0.0/data/en/silver/p80/d0541/en.drs.sbn`: `of developed.v.01 `
* `pmb-4.0.0/data/en/bronze/p60/d0288/en.drs.sbn`: `2 million-dollar.a.01`

- In it/de/nl there are more sense ids with spaces in them, but again, not a lot:
   * IT: 3 / 106737
   * DE: 22 / 160692
   * NL: 23 / 31172
