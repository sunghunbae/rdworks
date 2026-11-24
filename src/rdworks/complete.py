"""Backward-compatible aliases.

The functions formerly defined here have moved:

- `complete_stereoisomers()`, `can_have_stereo()` -> `rdworks.stereoisomers`
- `complete_tautomers()` -> `rdworks.tautomers`
"""

from rdworks.stereoisomers import complete_stereoisomers, can_have_stereo  # noqa: F401
from rdworks.tautomers import complete_tautomers  # noqa: F401
