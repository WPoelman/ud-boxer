import re
from typing import List, Optional, Tuple

__all__ = ["SBNSpec", "split_comments"]


class SBNSpec:
    COMMENT = r" % "
    COMMENT_LINE = r"%%%"

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

    ROLES = {
        # Manually added (not part of clf_signature.yaml)
        "TSU",  # What does this mean?
        "MOR",
        "BOT",
        "TOP",
        "ESU",
        "EPR",
        "InstanceOf",  # what is the difference with "Instance"?
        # --- From here down copied from clf_signature.yaml ---
        # Concept roles
        "Proposition",
        "Name",
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
    # # Special constants at the 'ending' nodes
    # CONSTANTS = {
    #     "speaker",
    #     "hearer",
    #     "now",
    #     "unknown_ref",
    #     "monday",
    #     "tuesday",
    #     "wednesday",
    #     "thursday",
    #     "friday",
    #     "saturday",
    #     "sunday",
    # }

    # # "'2008'" or "'196X'"" for instance, always in single quotes
    # YEAR_CONSTANT = r"\'([\dX]{4})\'"

    # # Can be "?", single "+/-" or unsigned digit (explicitly signed digits are
    # # assumed to be indices and are matched first!)
    # QUANTITY_CONSTANT = r"[\+\-\d\?]"

    # # "Tom got an A on his exam": Value -> "A" NOTE: arguably better to catch
    # # this with roles, but all other constants are caught.
    # VALUE_CONSTANT = r"^[A-Z]$"

    # CONSTANTS_PATTERN = re.compile(
    #     "|".join([YEAR_CONSTANT, QUANTITY_CONSTANT, VALUE_CONSTANT, CONSTANTS])
    # )

    MIN_SENSE_IDX = 0


def split_comments(sbn_string: str) -> List[Tuple[str, Optional[str]]]:
    """
    Helper to remove starting comments and split the actual sbn and
    trailing comments per line. Empty comments are converted to None.
    """
    # First split everything in lines
    split_lines = sbn_string.rstrip("\n").split("\n")

    # Separate the actual SBN and the comments
    temp_lines = []
    for line in split_lines:
        # discarded here.
        if line.startswith(SBNSpec.COMMENT_LINE):
            continue

        # Split lines in (<SBN-line>, <comment>) and filter out empty comments.
        items = line.split(SBNSpec.COMMENT, 1)

        sbn, comment = [item.strip() for item in items]
        temp_lines.append((sbn, comment or None))

    return temp_lines
