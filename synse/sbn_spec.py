"""
This SBN spec is based on the PMB 4.0.0 .
There are some know issues with SBN that are being looked at:
    - A constant that looks like an index and *is* a valid index cannot be
      distinguished from a regular index currently. So with the sentence
      "The temperature was -1 degrees yesterday", -1 will be interpreted as an
      index and not a constant.
    - There are some sense id's that contain whitespace. This gives problems
      for the tokenization. All examples can be found in 
      'data/misc/whitespace_in_ids.txt' of this repo.
"""

import re
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "SBNError",
    "SBNSpec",
    "split_comments",
    "split_wn_sense",
    "get_doc_id",
]


class SBNError(Exception):
    pass


class SBNSpec:
    COMMENT = r" % "
    COMMENT_LINE = r"%%%"
    MIN_SENSE_IDX = 0

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

    REVERSABLE_ROLES = {
        "Instance",
        "Attribute",
        "Colour",
        "Content",
        "Made",
        "Part",
        "Sub",
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
    WORDNET_SENSE_PATTERN = re.compile(r"(.+)\.(n|v|a|r)\.(\d+)")
    INDEX_PATTERN = re.compile(r"((-|\+)\d)")
    NAME_CONSTANT_PATTERN = re.compile(r"\"(.+)\"|\"(.+)")

    # NOTE: Now roles are properly handled instead of indirectly, but these
    # constant patterns might still be handy.

    # Special constants at the 'ending' nodes
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

    # CONSTANTS_PATTERN = re.compile(
    #     "|".join([YEAR_CONSTANT, QUANTITY_CONSTANT, VALUE_CONSTANT, CONSTANTS])
    # )


def split_comments(sbn_string: str) -> List[Tuple[str, Optional[str]]]:
    """
    Helper to remove starting comments and split the actual sbn and
    trailing comments per line. Empty comments are converted to None.
    """
    # First split everything in lines
    split_lines = sbn_string.rstrip("\n").split("\n")

    # Separate the actual SBN and the comments
    temp_lines: List[Tuple[str, Optional[str]]] = []
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
            temp_lines.append((items[0], items[1]))
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
        if SBNSpec.WORDNET_SENSE_PATTERN.match(token) or (
            token in SBNSpec.NEW_BOX_INDICATORS
        ):
            token = f"\n{token}"
        final_tokens.append(token)

    final_string = " ".join(final_tokens).strip()
    return final_string


def split_wn_sense(sense_id: str) -> Optional[Tuple[str, str, str]]:
    """
    Splits a wordnet sense into its components: lemma, pos, sense_number.
    """
    if match := SBNSpec.WORDNET_SENSE_PATTERN.match(sense_id):
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
