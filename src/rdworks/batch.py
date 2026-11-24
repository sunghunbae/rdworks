"""Batches of conformers for batch calculators.

`prepare_batches()` only reads `conf.natoms` and `conf.rdmol`, so this module does not
import `rdworks.conf`; both `rdworks.conf` and `rdworks.torsion` import it.
"""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdworks.conf import Conf


Batch = namedtuple("Batch", ["rdmols", "size", "num_atoms"])


def prepare_batches(confs: list[Conf], batchsize_atoms: int = 16384) -> list[Batch]:
    """Prepare batches of Chem.Mol conformers for batch optimization.

    - Each batch has up to `batchsize_atoms` number of atoms.
    - Each batch (namedtuple) has `rdmols`, `size`, `num_atoms` attributes.
    - Note that this function is not necessary for torsion energies
        the batch-optimization of torsion energies are performed using internally prepared batches
        of a series of conformers with different dihedral angles.

    Args:
        confs (list[Conf]): confs to be prepared for batch processing.
        batchsize_atoms (int): max number of atoms in one batch. Defaults to 16384(=16*1024).

    Returns:
        list[namedtuple]: list of batches.
            [Batch(rdmols=batch_confs, size=len(batch_confs), num_atoms=batch_atoms),...]
    """
    batches = []
    batch_confs = []
    batch_atoms = 0
    for conf in confs:
        if batch_confs and (batch_atoms + conf.natoms) > batchsize_atoms:
            batches.append(
                Batch(rdmols=batch_confs, size=len(batch_confs), num_atoms=batch_atoms)
            )
            # start over a new batch (a conf larger than batchsize_atoms gets its own)
            batch_confs = [conf.rdmol]
            batch_atoms = conf.natoms
        else:
            batch_confs.append(conf.rdmol)
            batch_atoms += conf.natoms
    if batch_atoms > 0:  # last remaining batch
        batches.append(
            Batch(rdmols=batch_confs, size=len(batch_confs), num_atoms=batch_atoms)
        )

    return batches
