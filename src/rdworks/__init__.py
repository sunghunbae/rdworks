import logging
from importlib.metadata import PackageNotFoundError, version

from rdworks.conf import Conf
from rdworks.mol import Mol
from rdworks.mollibr import MolLibr
from rdworks.microstate import State, StateEnsemble, StateNetwork
from rdworks.matchedseries import MatchedSeries

import rdkit

try:
    __version__ = version("rdworks")
except PackageNotFoundError:
    __version__ = "unknown"

__rdkit_version__ = rdkit.rdBase.rdkitVersion

# Library logging best practice: attach only a NullHandler and leave levels,
# handlers and formatting to the application. RDKit's own log level is also
# left to the user, e.g. ``from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")``.
# To see rdworks messages: ``logging.basicConfig(level=logging.INFO)``.
logging.getLogger(__name__).addHandler(logging.NullHandler())