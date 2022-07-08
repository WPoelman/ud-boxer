Apparently sometimes a 'person' is split up into the person and
their role, another one that could be expanded possibly.

Example: `pmb-4.0.0/data/en/gold/p35/d2181`

---

A 'compound' can also combine into a single node. Need to check if
this is always the case or not:

Example: `pmb-4.0.0/data/en/gold/p35/d1933`

---

Units need to be split up as well into
measure
  -> quantity
  -> unit

Example: `pmb-4.0.0/data/en/gold/p13/d2548`

---

The root is not always a verb with a tense, so the time node is
not guaranteed, instead try all nodes and do a majority vote
on the Tense features (or look at pos?). Default is 'now'

Example: `pmb-4.0.0/data/en/gold/p13/d2559` (ADJ as root, which is a bit strange)

---

advmod -> manner ?

Example: `pmb-4.0.0/data/en/gold/p45/d1392`

---

nmod:poss & subj is ambiguous on a node - edge level, pos does not help here.

Example: `pmb-4.0.0/data/en/gold/p04/d1646` -> User

Example: `pmb-4.0.0/data/en/gold/p42/d2737` -> PartOf


---

Multiple time nodes have to be connected to the same time node
(probably not always true, but for now).

Example: `pmb-4.0.0/data/en/gold/p53/d0764`

---

mark deprel + sconj -> new box? Explanation?

Example: `pmb-4.0.0/data/en/gold/p00/d0801` (in test cases)

---

advmod deprel + part -> new box negation?

Example: `pmb-4.0.0/data/en/gold/p00/d1593` (in test cases)

---

xcomp edges can combine in an attribute of sorts, multiple
xcomp nodes might need to be combined.

Example: `pmb-4.0.0/data/en/gold/p00/d2719` (in test cases)

---

need to add more options for combining compounds mixed with
flats (and the other way around) or combine compounds at a later
stage regardless of the flats?

Example: `pmb-4.0.0/data/en/gold/p03/d2003` (in test cases)

---

You cannot differentiate compounds based on NOUN -[compound]-> NOUN, wether they need to become a single synset or an adjective-like role (example: en/gold/p05/d2113). The same goes for proper adjectives in most cases (colors get combined).
