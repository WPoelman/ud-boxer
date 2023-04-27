"""
This SBN spec is based on the PMB 4.0.0 .
There are some know issues with SBN that are being looked at:
    - A constant that looks like an index and *is* a valid index cannot be
      distinguished from a regular index currently. So with the sentence
      "The temperature was -1 degrees yesterday", -1 will be interpreted as an
      index and not a constant.
    - There are some synset id's that contain whitespace. This gives problems
      for the tokenization. All examples can be found in 
      'data/misc/whitespace_in_ids.txt' of this repo.
"""

import re
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple

from ud_boxer.base import BaseEnum

__all__ = [
    "SBN_NODE_TYPE",
    "SBN_EDGE_TYPE",
    "SBNError",
    "AlignmentError",
    "SBNSpec",
    "split_comments",
    "split_synset_id",
    "get_doc_id",
]


class SBN_NODE_TYPE(BaseEnum):
    """Node types"""

    SYNSET = "synset"
    CONSTANT = "constant"
    BOX = "box"


class SBN_EDGE_TYPE(BaseEnum):
    """Edge types"""

    ROLE = "role"
    DRS_OPERATOR = "drs-operator"
    BOX_CONNECT = "box-connect"
    BOX_BOX_CONNECT = "box-box-connect"


class SBNError(Exception):
    pass

class AlignmentError(Exception):
    pass
class SBNSpec:
    COMMENT = r" % "
    COMMENT_LINE = r"%%%"
    MIN_SYNSET_IDX = 0

    DOC_ID_PATTERN = re.compile(r"(p\d{2}/d\d{4})")

    NEW_BOX_INDICATORS = {
        "ALTERNATION",
        "ATTRIBUTION",
        "CONDITION",
        "CONSEQUENCE",
        "CONTINUATION",
        "CONTRAST",
        "EXPLANATION",
        "NECESSITY",
        "NEGATION",
        "POSSIBILITY",
        "PRECONDITION",
        "RESULT",
        "SOURCE",
    }
    NEW_BOX_INDICATORS_2VERB = {
        "NECESSITY",
        "POSSIBILITY",
        "CONDITION",
        'NEGATION'
    }

    NEW_BOX_INDICATORS_VERB2VERB = {
        "ALTERNATION",
        "ATTRIBUTION",
        "CONSEQUENCE",
        "CONTINUATION",
        "CONTRAST",
        "EXPLANATION",
        "RESULT",
        "SOURCE",

    }
    DRS_OPERATORS = {
        # Manually added (not part of clf_signature.yaml)
        "TSU",  # What does this mean?
        "MOR",
        "BOT",
        "TOP",
        "ESU",
        "EPR",
        # --- From here down copied from clf_signature.yaml ---
        # temporal relations
        "EQU",  # equal
        "NEQ",  # not equla
        "APX",  # approximate
        "LES",  # less
        "LEQ",  # less or equal
        "TPR",  # precede
        "TAB",  # abut
        "TIN",  # include
        # spatial operators
        "SZP",  # above x / y
        "SZN",  # under x \ y
        "SXP",  # behind x » y
        "SXN",  # before x « y
        "STI",  # inside
        "STO",  # outside
        "SY1",  # beside
        "SY2",  # between
        "SXY",  # around
    }

    INVERTIBLE_ROLES = {
        "InstanceOf",
        "AttributeOf",
        "ColourOf",
        "ContentOf",
        "PartOf",
        "SubOf",
    }

    ROLES = {
        # Manually added (not part of clf_signature.yaml)
        "InstanceOf",
        # --- From here down copied from clf_signature.yaml ---
        # Concept roles
        "Proposition",
        "Name",
        # Event roles
        "Agent",
        "Asset",
        "Attribute",
        "AttributeOf",
        "Beneficiary",
        "Causer",
        "Co-Agent",
        "Co-Patient",
        "Co-Theme",
        "Consumer",
        "Destination",
        "Duration",
        "Experiencer",
        "Finish",
        "Frequency",
        "Goal",
        "Instrument",
        "Instance",
        "Location",
        "Manner",
        "Material",
        "Path",
        "Patient",
        "Pivot",
        "Product",
        "Recipient",
        "Result",
        "Source",
        "Start",
        "Stimulus",
        "Theme",
        "Time",
        "Topic",
        "Value",
        # Concept roles
        "Bearer",
        "Colour",
        "ColourOf",
        "ContentOf",
        "Content",
        "Creator",
        "Degree",
        "MadeOf",
        # - Name
        "Of",
        "Operand",
        "Owner",
        "Part",
        "PartOf",
        "Player",
        "Quantity",
        "Role",
        "Sub",
        "SubOf",
        "Title",
        "Unit",
        "User",
        # Time roles
        "ClockTime",
        "DayOfMonth",
        "DayOfWeek",
        "Decade",
        "MonthOfYear",
        "YearOfCentury",
        # Other roles.
        "Affector",
        "Context",
        "Equal",
        "Extent",
        "Precondition",
        "Measure",
        "Cause",
        "Order",
        "Participant",
    }

    # The lemma match might seem loose, however there can be a lot of different
    # characters in there: 'r/2.n.01', 'ø.a.01', 'josé_maria_aznar.n.01'
    SYNSET_PATTERN = re.compile(r"(.+)\.(n|v|a|r)\.(\d+)")
    INDEX_PATTERN = re.compile(r"((-|\+)\d)")
    NAME_CONSTANT_PATTERN = re.compile(r"\"(.+)\"|\"(.+)")

    # NOTE: Now roles are properly handled instead of indirectly, but these
    # constant patterns might still be handy.

    # Special constants at the 'ending' (leaf) nodes
    CONSTANTS = {
        "speaker",
        "hearer",
        "now",
        "unknown_ref",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }

    # "'2008'" or "'196X'"" for instance, always in single quotes
    YEAR_CONSTANT = r"\'([\dX]{4})\'"

    # Can be "?", single "+/-" or unsigned digit (explicitly signed digits are
    # assumed to be indices and are matched first!)
    QUANTITY_CONSTANT = r"[\+\-\d\?]"

    # "Tom got an A on his exam": Value -> "A"
    VALUE_CONSTANT = r"^[A-Z]$"


def split_comments(sbn_string: str) -> List[Tuple[str, Optional[str]]]:
    """
    Helper to remove starting comments and split the actual sbn and
    trailing comments per line. Empty comments are converted to None.
    """
    # First split everything in lines
    split_lines = sbn_string.rstrip("\n").split("\n")

    # Separate the actual SBN and the comments
    temp_lines: List[Tuple[str, Optional[str]]] = []
    count =-1
    for line in split_lines:
        # Full comment lines are discarded here.
        if line.startswith(SBNSpec.COMMENT_LINE):
            continue

        # Split lines in (<SBN-line>, <comment>) and filter out empty comments.
        items = []
        for item in line.split(SBNSpec.COMMENT, 1):
            if item := item.strip():
                items.append(item)

        # An empty line
        if len(items) == 0:
            pass
        # There is no comment
        elif len(items) == 1:
            temp_lines.append((items[0], None))
        # We have a comment
        elif len(items) == 2:
            count+=1
            temp_lines.append((items[0], items[1].split('[')[0].strip()+str(count)))
        else:
            raise SBNError(
                "Unreachable, multiple comments per line are impossible"
            )

    return temp_lines


def split_single(sbn_string: str) -> str:
    """
    Helper to convert SBN that is in a single, flat string (no newlines) into
    separate lines.
    """
    tokens = sbn_string.split(" ")
    final_tokens = []

    for token in tokens:
        if SBNSpec.SYNSET_PATTERN.match(token) or (
            token in SBNSpec.NEW_BOX_INDICATORS
        ):
            token = f"\n{token}"
        final_tokens.append(token)

    final_string = " ".join(final_tokens).strip()
    return final_string


def split_synset_id(syn_id: str) -> Optional[Tuple[str, str, str]]:
    """
    Splits a WordNet synset into its components: lemma, pos, sense_number.
    """
    if match := SBNSpec.SYNSET_PATTERN.match(syn_id):
        return (match.group(1), match.group(2), match.group(3))
    return None


def get_doc_id(lang: str, filepath: PathLike) -> str:
    """
    Helper to extract a doc id from the filepath of the sbn file.
    A doc id has the format <lang>/<p>/<d>
    """
    return f"{lang}/{get_base_id(filepath)}"


def get_base_id(filepath: PathLike) -> str:
    """
    Helper to extract a doc id from the filepath of the sbn file.
    A base doc id has the format <p>/<d>
    """
    full_path = str(Path(filepath))
    if match := SBNSpec.DOC_ID_PATTERN.search(full_path):
        return match.group(1)

    raise SBNError("Could not extract doc id!")
