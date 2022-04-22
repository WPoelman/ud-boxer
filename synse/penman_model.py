"""
Adapted from https://github.com/goodmami/penman/blob/main/penman/models/amr.py
This adaptation for SBN (DRS) is based on the `sbn_spec.py`.
"""

from typing import Any, List, Mapping

from penman.model import Model, _ReificationSpec
from penman.types import Role

from synse.sbn_spec import SBNSpec

__all__ = [
    "pm_model",
]

# TODO: figure out if inverting certain roles can be done with the reifications
# or normalizations.

#: The roles are the edge labels of reifications. The purpose of roles
#: in a :class:`~penman.model.Model` is mainly to define the set of
#: valid roles, but they map to arbitrary data which is not used by
#: the :class:`~penman.model.Model` but may be inspected or used by
#: client code.
roles: Mapping[Role, Any] = {
    # Regular SBN spec items
    **{f":{k}": dict(token=k) for k in SBNSpec.NEW_BOX_INDICATORS},
    **{f":{k}": dict(token=k) for k in SBNSpec.DRS_OPERATORS},
    **{f":{k}": dict(token=k) for k in SBNSpec.ROLES},
    # Custom extra rules for (lenient) penman output
    **{f":{k}": dict(token=k) for k in ["lemma", "pos", "sense"]},
}


#: Normalizations are like role aliases. If the left side of the
#: normalization is encountered by
#: :meth:`penman.model.Model.canonicalize_role` then it is replaced
#: with the right side.
normalizations: Mapping[Role, Role] = dict()
# ":mod-of":    ":domain",


#: Reifications are a particular kind of transformation that replaces
#: an edge relation with a new node and two outgoing edge relations,
#: one inverted. They are used when the edge needs to behave as a node,
#: e.g., to be modified or focused.
reifications: List[_ReificationSpec] = []
# role           concept                source   target
# (":accompanier", "accompany-01",        ":ARG0", ":ARG1"),


pm_model = Model(
    top_variable="b0",
    top_role=":member",
    concept_role=":instance",
    roles=roles,
    normalizations=normalizations,
    reifications=reifications,
)
