__all__ = [
    "UDSpecBasic",
    "UPOS_WN_POS_MAPPING",
    "TIME_EDGE_MAPPING",
    "GENDER_SENSE_MAPPING",
]


class UDSpecBasic:
    class DepRels:
        ACL = "acl"  # clausal modifier of noun (adnominal clause)
        ACL_RELCL = "acl:relcl"  # relative clause modifier
        ADVCL = "advcl"  # adverbial clause modifier
        ADVMOD = "advmod"  # adverbial modifier
        ADVMOD_EMPH = "advmod:emph"  # emphasizing word, intensifier
        ADVMOD_LMOD = "advmod:lmod"  # locative adverbial modifier
        AMOD = "amod"  # adjectival modifier
        APPOS = "appos"  # appositional modifier
        AUX = "aux"  # auxiliary
        AUX_PASS = "aux:pass"  # passive auxiliary
        CASE = "case"  # case marking
        CC = "cc"  # coordinating conjunction
        CC_PRECONJ = "cc:preconj"  # preconjunct
        CCOMP = "ccomp"  # clausal complement
        CLF = "clf"  # classifier
        COMPOUND = "compound"  # compound
        COMPOUND_LVC = "compound:lvc"  # light verb construction
        COMPOUND_PRT = "compound:prt"  # phrasal verb particle
        COMPOUND_REDUP = "compound:redup"  # reduplicated compounds
        COMPOUND_SVC = "compound:svc"  # serial verb compounds
        CONJ = "conj"  # conjunct
        COP = "cop"  # copula
        CSUBJ = "csubj"  # clausal subject
        CSUBJ_PASS = "csubj:pass"  # clausal passive subject
        DEP = "dep"  # unspecified dependency
        DET = "det"  # determiner
        DET_NUMGOV = "det:numgov"  # pronominal quantifier governing the case of the noun
        DET_NUMMOD = "det:nummod"  # pronominal quantifier agreeing in case with the noun
        DET_PREDET = "det:predet"  # relation between the head of NP and a word that precedes and modifies the meaning of the NP determiner. "all the boys" all < boys
        DET_POSS = "det:poss"  # possessive determiner
        DISCOURSE = "discourse"  # discourse element
        DISLOCATED = "dislocated"  # dislocated elements
        EXPL = "expl"  # expletive
        EXPL_IMPERS = "expl:impers"  # impersonal expletive
        EXPL_PASS = "expl:pass"  # reflexive pronoun used in reflexive passive
        EXPL_PV = (
            "expl:pv"  # reflexive clitic with an inherently reflexive verb
        )
        FIXED = "fixed"  # fixed multiword expression
        FLAT = "flat"  # flat multiword expression
        FLAT_FOREIGN = "flat:foreign"  # foreign words
        FLAT_NAME = "flat:name"  # names
        GOESWITH = "goeswith"  # goes with
        IOBJ = "iobj"  # indirect object
        LIST = "list"  # list
        MARK = "mark"  # marker
        NMOD = "nmod"  # nominal modifier
        NMOD_NPMOD = "nmod:npmod"  # something syntactically a noun phrase is used as an adverbial modifier in a sentence (see obl:npmod)
        NMOD_POSS = "nmod:poss"  # possessive nominal modifier
        NMOD_TMOD = "nmod:tmod"  # temporal modifier
        NSUBJ = "nsubj"  # nominal subject
        NSUBJ_PASS = "nsubj:pass"  # passive nominal subject
        NUMMOD = "nummod"  # numeric modifier
        NUMMOD_GOV = (
            "nummod:gov"  # numeric modifier governing the case of the noun
        )
        OBJ = "obj"  # object
        OBL = "obl"  # oblique nominal
        OBL_AGENT = "obl:agent"  # agent modifier
        OBL_ARG = "obl:arg"  # oblique argument
        OBL_LMOD = "obl:lmod"  # locative modifier
        OBL_TMOD = "obl:tmod"  # temporal modifier
        OBL_NPMOD = "obl:npmod"  # noun phrase used as an adverbial modifier "65 years < old"
        ORPHAN = "orphan"  # orphan
        PARATAXIS = "parataxis"  # parataxis
        PUNCT = "punct"  # punctuation
        REPARANDUM = "reparandum"  # overridden disfluency
        ROOT = "root"  # root
        VOCATIVE = "vocative"  # vocative
        XCOMP = "xcomp"  # open clausal complement

        ALL_DEP_RELS = {
            ACL,
            ACL_RELCL,
            ADVCL,
            ADVMOD,
            ADVMOD_EMPH,
            ADVMOD_LMOD,
            AMOD,
            APPOS,
            AUX,
            AUX_PASS,
            CASE,
            CC,
            CC_PRECONJ,
            CCOMP,
            CLF,
            COMPOUND,
            COMPOUND_LVC,
            COMPOUND_PRT,
            COMPOUND_REDUP,
            COMPOUND_SVC,
            CONJ,
            COP,
            CSUBJ,
            CSUBJ_PASS,
            DEP,
            DET,
            DET_NUMGOV,
            DET_NUMMOD,
            DET_PREDET,
            DET_POSS,
            DISCOURSE,
            DISLOCATED,
            EXPL,
            EXPL_IMPERS,
            EXPL_PASS,
            EXPL_PV,
            FIXED,
            FLAT,
            FLAT_FOREIGN,
            FLAT_NAME,
            GOESWITH,
            IOBJ,
            LIST,
            MARK,
            NMOD,
            NMOD_NPMOD,
            NMOD_POSS,
            NMOD_TMOD,
            NSUBJ,
            NSUBJ_PASS,
            NUMMOD,
            NUMMOD_GOV,
            OBJ,
            OBL,
            OBL_AGENT,
            OBL_ARG,
            OBL_LMOD,
            OBL_TMOD,
            OBL_NPMOD,
            ORPHAN,
            PARATAXIS,
            PUNCT,
            REPARANDUM,
            ROOT,
            VOCATIVE,
            XCOMP,
        }

    class POS:
        ADJ = "ADJ"  # adjective
        ADP = "ADP"  # adposition
        ADV = "ADV"  # adverb
        AUX = "AUX"  # auxiliary
        CCONJ = "CCONJ"  # coordinating conjunction
        DET = "DET"  # determiner
        INTJ = "INTJ"  # interjection
        NOUN = "NOUN"  # noun
        NUM = "NUM"  # numeral
        PART = "PART"  # particle
        PRON = "PRON"  # pronoun
        PROPN = "PROPN"  # proper noun
        PUNCT = "PUNCT"  # punctuation
        SCONJ = "SCONJ"  # subordinating conjunction
        SYM = "SYM"  # symbol
        VERB = "VERB"  # verb
        X = "X"  # other

        ALL_POS = {
            ADJ,
            ADP,
            ADV,
            AUX,
            CCONJ,
            DET,
            INTJ,
            NOUN,
            NUM,
            PART,
            PRON,
            PROPN,
            PUNCT,
            SCONJ,
            SYM,
            VERB,
            X,
        }

    class Feats:
        KEYS = {
            "Abbr",
            "Animacy",
            "Aspect",
            "Case",
            "Clusivity",
            "Definite",
            "Degree",
            "Evident",
            "Foreign",
            "Gender",
            "Mood",
            "NounClass",
            "Number",
            "NumType",
            "Person",
            "Polarity",
            "Polite",
            "Poss",
            "PronType",
            "Reflex",
            "Tense",
            "Typo",
            "VerbForm",
            "Voice",
        }

        class Tense:
            # https://universaldependencies.org/u/feat/Tense.html
            FUT = "Fut"
            IMP = "Imp"
            PAST = "Past"
            PQP = "Pqp"
            PRES = "Pres"

        class Gender:
            # https://universaldependencies.org/u/feat/Gender.html
            COM = "Com"
            FEM = "Fem"
            MASC = "Masc"
            NEUT = "Neut"


# Wordnet has the following pos tags: nouns, verbs, adjectives and adverbs.
# These are indicated by n, v, a and r respectively. The default mapping is to
# a noun! These mappings are by no means perfect, but serve as a starting
# point and to allow further processing later on.
UPOS_WN_POS_MAPPING = {
    UDSpecBasic.POS.ADJ: "a",
    UDSpecBasic.POS.ADP: "n",
    UDSpecBasic.POS.ADV: "r",
    UDSpecBasic.POS.AUX: "v",
    UDSpecBasic.POS.CCONJ: "n",
    UDSpecBasic.POS.DET: "n",
    UDSpecBasic.POS.INTJ: "n",
    UDSpecBasic.POS.NOUN: "n",
    UDSpecBasic.POS.NUM: "n",
    UDSpecBasic.POS.PART: "r",  # difficult one: https://universaldependencies.org/u/pos/PART.html
    UDSpecBasic.POS.PRON: "n",
    UDSpecBasic.POS.PROPN: "n",
    UDSpecBasic.POS.PUNCT: "n",
    UDSpecBasic.POS.SCONJ: "n",
    UDSpecBasic.POS.SYM: "n",
    UDSpecBasic.POS.VERB: "v",
    UDSpecBasic.POS.X: "n",
}

TIME_EDGE_MAPPING = {
    UDSpecBasic.Feats.Tense.FUT: "TSU",
    UDSpecBasic.Feats.Tense.IMP: "TPR",  # Not in English?
    UDSpecBasic.Feats.Tense.PAST: "TPR",
    UDSpecBasic.Feats.Tense.PQP: "TPR",  # Not in English
    UDSpecBasic.Feats.Tense.PRES: "EQU",  # or TIN (it's still happening)
}

GENDER_SENSE_MAPPING = {
    UDSpecBasic.Feats.Gender.COM: "person.n.01",
    UDSpecBasic.Feats.Gender.FEM: "female.n.02",
    UDSpecBasic.Feats.Gender.MASC: "male.n.02",
    UDSpecBasic.Feats.Gender.NEUT: "person.n.01",
}


class UDSpecExtended:
    # NOTE: not supported yet
    pass
