from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


FP_TYPES = ("morgan", "rdkit", "maccs")


def distance_function(i: int, j: int, fps: list) -> float:
    """Calculate Tanimoto distance between two fingerprints."""
    return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])


def calculate_fingerprints(
    molecules: list, fp_type: str = "morgan", radius: int = 2, n_bits: int = 2048
) -> list:
    """Calculate molecular fingerprints for a list of molecules.

    Args:
        molecules (list): rdkit.Chem.Mol or rdworks.Mol objects (e.g. `MolLibr.libr`).
            None entries give None fingerprints.
        fp_type (str): type of fingerprint: 'morgan', 'rdkit', or 'maccs'.
            Defaults to 'morgan'.
        radius (int): radius for Morgan fingerprints. Defaults to 2.
        n_bits (int): number of bits for Morgan and RDKit fingerprints
            (MACCS keys always have 167 bits). Defaults to 2048.

    Raises:
        ValueError: if `fp_type` is unknown.

    Returns:
        list: fingerprint bit vectors (or None).
    """
    if fp_type not in FP_TYPES:
        raise ValueError(f"Unknown fingerprint type: {fp_type}; supported: {FP_TYPES}")

    if fp_type == "morgan":
        # same bits as the deprecated AllChem.GetMorganFingerprintAsBitVect()
        morgan = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fps = []

    for mol in molecules:
        # accept rdworks.Mol (has .rdmol) as well as rdkit.Chem.Mol
        rdmol = getattr(mol, "rdmol", mol)
        if rdmol is None:
            fps.append(None)
            continue

        if fp_type == "morgan":
            fp = morgan.GetFingerprint(rdmol)
        elif fp_type == "rdkit":
            fp = Chem.RDKFingerprint(rdmol, fpSize=n_bits)
        else:  # maccs
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(rdmol)

        fps.append(fp)

    return fps
