from os import PathLike
from pathlib import Path
from typing import Set

from stanza.utils.conll import CoNLL

from synse.base import BaseEnum, BaseGraph
from synse.config import Config
from synse.ud_spec import UDSpecBasic

__all__ = [
    "UD_NODE_TYPE",
    "UD_EDGE_TYPE",
    "UD_SYSTEM",
    "UDError",
    "UDGraph",
    "UDParser",
    "Collector",
]


class UDError(Exception):
    pass


class UD_NODE_TYPE(BaseEnum):
    """Node types"""

    SENTENCE = "sentence"
    TOKEN = "token"
    ROOT = "root"


class UD_EDGE_TYPE(BaseEnum):
    """Edge types"""

    SENTENCE_CONNECT = "sentence-connect"
    DEPENDENCY_RELATION = "dependency-relation"
    EXPLICIT_ROOT = "explicit-root"


class UDGraph(BaseGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        # NOTE: detect UDSpec here once that is supported or make it an
        # argument. (enhanced UD vs standard/basic UD)
        self.root_node_ids = []

    def from_path(self, conll_path: PathLike):
        """Construct the graph using the provided conll file"""
        sentences, _ = CoNLL.conll2dict(conll_path)

        nodes, edges = [], []
        for sentence_idx, sentence in enumerate(sentences):
            # Explicitly add the root node for each sentence
            root_id = (sentence_idx, UD_NODE_TYPE.ROOT, 0)
            nodes.append(
                (
                    root_id,
                    {
                        "_id": root_id,
                        "token": "ROOT",
                        "lemma": None,
                        "deprel": None,
                        "upos": None,
                        "xpos": None,
                        "feats": None,
                        "connl_id": None,
                        "type": UD_NODE_TYPE.ROOT,
                    },
                )
            )
            if sentence_idx > 0:
                edges.append(
                    (
                        (sentence_idx - 1, UD_NODE_TYPE.ROOT, 0),
                        root_id,
                        {
                            "token": "new-sentence",
                            "deprel": None,
                            "type": UD_EDGE_TYPE.SENTENCE_CONNECT,
                        },
                    )
                )

            for token in sentence:
                # Ids are read in as tuples, but currently there are no parses
                # with multiple or duplicate ids (not sure when that happens,
                # with pre-annotated docs maybe?)
                if len(token["id"]) != 1:
                    raise UDError(
                        f"Multiple ids found, cannot parse this currently."
                    )

                tok_id = (sentence_idx, UD_NODE_TYPE.TOKEN, token["id"][0])
                dep_rel = token.get("deprel")
                upos = token.get("upos")

                if dep_rel not in UDSpecBasic.DepRels.ALL_DEP_RELS:
                    raise UDError(f"Unknown deprel found {dep_rel}")
                if upos not in UDSpecBasic.POS.ALL_POS:
                    raise UDError(f"Unknown upos found {upos}")

                # Morphological features are optional (can be None) and are
                # encoded as follows:
                #   <feat_1_key>=<feat_1_val>|<feat_2_key>=<feat_2_val> ...
                # So for instance: Mood=Ind|Tense=Past|VerbForm=Fin
                # None of the features are required on a token level.
                if feats_str := token.get("feats"):
                    feats = dict()
                    for key, value in [
                        item.split("=") for item in feats_str.split("|")
                    ]:
                        if key not in UDSpecBasic.Feats.KEYS:
                            raise UDError(f"Unknown Feat key found: {key}")
                        feats[key] = value

                tok_data = {
                    "_id": tok_id,
                    "token": token["text"],
                    "lemma": token.get("lemma"),
                    "deprel": dep_rel,
                    "upos": upos,
                    "xpos": token.get("xpos"),
                    "feats": feats or dict(),
                    "connl_id": token.get("id"),
                    "type": tok_id[1],
                }

                if token["head"] == 0:
                    head_id = (sentence_idx, UD_NODE_TYPE.ROOT, token["head"])
                    edge_type = UD_EDGE_TYPE.EXPLICIT_ROOT
                    self.root_node_ids.append(tok_id)
                else:
                    head_id = (sentence_idx, UD_NODE_TYPE.TOKEN, token["head"])
                    edge_type = UD_EDGE_TYPE.DEPENDENCY_RELATION

                edge_data = {
                    "token": token["deprel"],
                    "deprel": token["deprel"],
                    "type": edge_type,
                }

                nodes.append((tok_id, tok_data))
                edges.append((head_id, tok_id, edge_data))

        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        return self

    def root_node(self, sentence_idx: int = 0):
        return self.nodes[self.root_node_ids[sentence_idx]]

    @staticmethod
    def _node_label(node_data) -> str:
        label = [node_data["token"]]

        if lemma := node_data.get("lemma"):
            label.append(lemma)
        if upos := node_data.get("upos"):
            label.append(upos)
        if xpos := node_data.get("xpos"):
            label.append(xpos)

        return "\n".join(label)

    @staticmethod
    def _edge_label(edge_data) -> str:
        return edge_data["token"]

    @property
    def type_style_mapping(self):
        return {
            UD_NODE_TYPE.SENTENCE: {},
            UD_NODE_TYPE.TOKEN: {},
            UD_NODE_TYPE.ROOT: {"shape": "box"},
            UD_EDGE_TYPE.SENTENCE_CONNECT: {"style": "dotted"},
            UD_EDGE_TYPE.DEPENDENCY_RELATION: {},
            UD_EDGE_TYPE.EXPLICIT_ROOT: {},
        }


class UDParser:
    def __init__(
        self,
        system: Config.UD_SYSTEM = Config.UD_SYSTEM.STANZA,
        language: Config.SUPPORTED_LANGUAGES = Config.SUPPORTED_LANGUAGES.EN,
    ) -> None:
        if system == Config.UD_SYSTEM.STANZA:
            from stanza import Pipeline, download
            from stanza.utils.conll import CoNLL

            # No need for very heavy NER / sentiment etc models currently
            processors = "tokenize,pos,lemma,depparse"
            download(language, processors=processors)
            pipeline = Pipeline(lang=language, processors=processors)

            def write_output(result, out_file):
                CoNLL.write_doc2conll(result, out_file)

        elif system == Config.UD_SYSTEM.TRANKIT:
            from trankit import Pipeline, trankit2conllu

            pipeline = Pipeline(Config.UD_LANG_MAPPING[language])

            def write_output(result, out_file):
                out_file.write_text(trankit2conllu(result))

        else:
            raise UDError(f"Unsupported UD_SYSTEM: {system}")

        self.pipeline = pipeline
        self.write_output = write_output

    def parse(self, text: str, out_file: PathLike) -> Path:
        """
        Generate a UD parse from the input text and store it in conll format
        at the provided path.
        """
        out_file = Path(out_file)
        result = self.pipeline(text)
        self.write_output(result, out_file)

        return out_file

    def parse_path(self, text_file: PathLike, out_file: PathLike) -> Path:
        """
        Generate a UD parse from input text file and store it in conll format
        at the provided path.
        """
        return self.parse(Path(text_file).read_text(), out_file)


class Collector:
    """Helper to collect some information about the UD graph"""

    def __init__(self) -> None:
        self.dep_rels: Set[str] = set()
        self.pos: Set[str] = set()

    def collect(self, U: UDGraph):
        # Quick and dirty way to check all used deprels and pos tags in the
        # dataset.
        self.dep_rels.update(
            {a[2]["deprel"] for a in U.edges.data() if a[2]["deprel"]}
        )
        self.pos.update({a[1]["xpos"] for a in U.nodes.data() if a[1]["xpos"]})
