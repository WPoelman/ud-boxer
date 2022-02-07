# thesis
Source for my master's thesis system for Information Science.

## Installation
1. (Optional) create and activate a virtual environment (tested with python 3.8+)
2. (Optional) for visualization install `graphviz`. 
   1. On Debian-based systems: `apt install graphviz libgraphviz-dev pkg-config`
   2. The `networkx` extras: `pip install networkx[default,extras]`
3. Install dependencies with `pip install -r requirements.txt`


## SBN output error: spaces in sense ids
There are a lot of ids with either a '-' or '_' connecting them and since there are only 3 in the entire dataset, this seems like an error.

* `pmb-4.0.0/data/en/silver/p92/d0079/en.drs.sbn`: `cocktails at.v.01`
* `pmb-4.0.0/data/en/silver/p80/d0541/en.drs.sbn`: `of developed.v.01 `
* `pmb-4.0.0/data/en/bronze/p60/d0288/en.drs.sbn`: `2 million-dollar.a.01`
