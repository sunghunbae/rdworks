# Change Log

## Unreleased
- added pixi workspace and conda package build (`pixi build`) configuration to `pyproject.toml`
- moved `complete_stereoisomers()` and `can_have_stereo()` to `stereoisomers.py` and `complete_tautomers()` to `tautomers.py` (`rdworks.complete` keeps backward-compatible aliases)
- fixed `can_have_stereo()` missing E/Z double bonds (e.g. 2-butene, imines, enols), which undercounted `complete_tautomers()` results
- fixed `complete_stereoisomers()` modifying the input `Mol` (name and stereochemistry)
- fixed `complete_tautomers()` dropping the name when a single result remains
- changed `make_confs()` in `Mol` to take explicit `calculator`, `water`, `batchsize_atoms`, `seed`, `prune_rms`, and `verbose` arguments
```py
m = m.make_confs(n=50, calculator='UFF', seed=42, prune_rms=0.5)
```
- fixed `make_confs()` crashing with `verbose=True` or `calculator=...`, unsupported `method` (e.g. `CONFORGE`), `n < 1`, or failed embedding
- changed `make_confs()` to fall back to UFF when MMFF94 has no parameters (e.g. B, Se, metals) and to keep existing conformers if it fails
- replaced shared `Mol.ETKDG_params` with `Mol.etkdg_params()`, which builds fresh multi-threaded ETKDGv3 parameters per call
- fixed `Conf.singlepoint()` raising an unclear `AttributeError` when a force field cannot be set up
- fixed `sort_confs()` crashing with `verbose=True` and a `calculator`
- fixed `drop_confs(cluster=True)` not dropping non-centroid conformers
- fixed `cluster_confs(method='DQT')` torsion difference wrapping and RMS normalization
- fixed `Mol.__hash__` to be consistent with `__eq__` (InChIKey), and `__eq__` with non-`Mol` objects
- fixed `is_matching()` ignoring descriptor bounds of 0 and stripping `~`/`!` beyond the leading marker
- removed broken `Mol.__next__` (iterate with `for conf in mol`)
- `align_confs()`, `draw()`, and `has_substr()` now raise `ValueError` for invalid input
- fixed `Mol()` from a 3D `Chem.Mol` with explicit hydrogens dropping a single conformer and crashing with multiple conformers (`clean_2d()` now returns one `Chem.Mol` per 3D conformer)
- fixed `desalt_smiles()` returning non-canonical SMILES, so a salt and its free form now give the same `Mol.smiles`
- fixed `desalt_smiles()` silently dropping invalid fragments and rejecting ring closures across `.` (e.g. `C1.C1`); invalid SMILES raise `ValueError`
- fixed `generate_inchi_key()` returning `None` when InChI is unavailable (e.g. dummy atoms); it returns `""` and `Mol` equality falls back to canonical SMILES
- `standardize()` now raises `ValueError` for invalid SMILES and `TypeError` for unsupported input
- fixed `SyntaxWarning` (invalid escape sequence) in `xtb/wrapper.py`
- fixed stale `Conf.positions` and `Conf.molblock` after `optimize()`, `sync()`, or `set_torsion_angle()`, which made ASE calculations, `serialize()` (including `Mol.serialize()`), `to_xyz()`, `to_geometry()`, and `to_turbomole()` use the old geometry; both are now properties computed from the current coordinates
- fixed `Conf.COG` and `Conf.Rg` always raising `AttributeError`
- fixed `Conf()` sharing the input `Chem.Mol` (coordinate changes moved the caller's molecule); the input is now copied and `name` is set on it
- `Conf()` now raises `ValueError` for an invalid MolBlock (previously not raised) and for input without exactly one conformer
- fixed `Conf.has_acceptable_bond_lengths()` always returning `False` by comparing with bond-order-adjusted covalent radii instead of van der Waals radii
- fixed `prepare_batches()` creating an empty batch when a conformer exceeds `batchsize_atoms`
- fixed `batch_singlepoint()` and `batch_optimize()` losing conformer names and props (stale energy and cluster props are dropped); a mismatched result count raises `RuntimeError`
- changed `str(Conf)` to show the SMILES instead of the object address
- fixed `calculate_fingerprints()` crashing on `rdworks.Mol` input; it now accepts `rdworks.Mol` and `rdkit.Chem.Mol`, and Morgan fingerprints use `rdFingerprintGenerator` (same bits as the deprecated `GetMorganFingerprintAsBitVect()`)
- fixed `MolLibr.pick_diverse()`, which always failed: it passed `rdworks.Mol` objects to the fingerprint function and a distance function to `LazyBitVectorPick()`, and would have kept the first `n` molecules instead of the picked ones
```py
libr = libr.pick_diverse(10, seed=0, fp_type='morgan')
```
- fixed `MolLibr` slicing (`libr[a:b]`) renaming the original library's molecules to `entry.N` and resetting compute settings
- fixed `MolLibr()` renaming the `rdworks.Mol` objects passed to it; only libraries with no named molecules are named `entry.1`, `entry.2`, ...
- fixed `MolLibr()` adding `None` for `rdworks.Conf` input and silently duplicating the previous molecule for unsupported input, which now raises `TypeError`
- fixed `MolLibr ==` returning `True` when one library is a subset of the other
- fixed `MolLibr` `-`, `-=`, `&`, and `&=` returning molecules in an arbitrary order that varied between Python sessions; results now keep the library order
- fixed `MolLibr.to_html()` always raising `AttributeError`; it embeds an SVG depiction and the name of each molecule
- fixed `MolLibr.to_batches()` creating an empty batch when a conformer exceeds `batchsize_atoms`; charges are now taken from each conformer
- removed broken `MolLibr.__next__` (iterate with `for mol in libr`)
- added shared writers `to_geometry()`, `to_xyz()`, `to_turbomole()`, `to_sdf()`, `to_png()`, `to_svg()`, `to_df()`, and `to_csv()` to `rdworks.io`; they accept `Conf`, `Mol`, `MolLibr`, or `rdkit.Chem.Mol`, and the `to_*` methods of `Conf`, `Mol`, and `MolLibr` now call them (existing calls keep working)
```py
import rdworks.io as rwio
rwio.to_xyz(libr, path='confs.xyz')   # same as libr.to_xyz(path='confs.xyz')
rwio.to_sdf(mol, confs=True)          # same as mol.to_sdf(confs=True)
```
- added `to_geometry()`, `to_xyz()` (all conformers), `to_turbomole()`, `to_df()`, and `to_csv()` to `Mol`; `to_xyz()` to `MolLibr`; and `to_df()` and `to_csv()` to `Conf`
- added an optional `path` to `to_xyz()`, `to_sdf()`, `to_png()`, and `to_svg()` of `Conf` and `Mol` and to `MolLibr.to_svg()`; `MolLibr.to_sdf()` returns SDF strings when `path` is omitted
- fixed `to_turbomole(bohr=True)` writing the `$coord` (Bohr) header with Angstrom values; coordinates are now converted (`units.angstrom2bohr` added)
- changed the XYZ comment line to the conformer name (was blank), so frames of multi-conformer XYZ files are identifiable
- changed `MolLibr.to_png(filename)` to also return the image after saving it
- changed `to_df()` column order to `name`, `smiles`, then properties in first-seen order (previously varied between Python sessions)
- `rdworks.io` no longer imports `Conf`, `Mol`, and `MolLibr` at module level (avoids a circular import); the readers import them when called

## 0.73.1 (Nov 25, 2025)
- created separate `batch.py`
- added concurrency to `singlepoint_confs()` in `Mol`
- added `calculate_sp_torsion_energies()` for single-point based torsion energies in `Mol` and `Conf`
- added concurrency to `optimize_confs()` in `Mol`
```py
m = m.make_confs(n=50).compute(max_workers=4, chunksize=2, progress=True)
m = m.optimize_confs(calculator='xTB', water='alpb')
```

## 0.72.7 (Nov 18, 2025)
- added `to_geometry()` to `Conf`
- added electrostatic potential surface calculation to `Conf` and `Mol`
- changed `get_geometric_SASA()` to a property `.SASA` in `Conf`
- changed `get_geometric_SASA()` to `get_SASA()` in `Mol`

## 0.71.3 (Nov 12, 2025)
- added `get_geometric_SASA()` to `Conf` and `Mol` (3D conformer-dependent solvent accessible surface area)
- added check a torsion angle atoms are in a ring for macrocycles at `get_torsion_angle_atom_indices()`
 
## 0.70.3 (Nov 7, 2025)
- fixed `get_charge_groups_per_site()`
  
## 0.70.2 (Nov 3, 2025)
- fixed micro- and macro-pKa calculations
- added micro- and macro-pKa tests

## 0.69.0 (Oct 27, 2025)
- added `DescriptiveDraw` class to `view.py`
- modified `Mol.align_confs()`, `Mol.cluster_confs()`, and `Mol.drop_confs()` for introducing spyrmsd symmetry-aware rmsd. 
- removed `altair` and `seaborn` dependencies

## 0.68.1 (Oct 20, 2025)
- added `pick_diverse()` to `MolLibr` class
- restructured `utils.py`
- renamed `display.py` as `view.py` and `readin.py` as `fileio.py`

## 0.67.3 (Oct 19, 2025)
- dictionary data are revised to have string key in `torsion_key` related functions to 
  avoid type change of key durin gJSON-serialization/deserialization.
  
## 0.67.2 (Oct 16, 2025)
- revised `serialize()` and `deserialize()` in `microstate.py`
- added `sort()` to `StateEnsemble` class to sort states by population(s) at given pH(s).
- revised `serialize()`, `deserialize()`, `compress_string()` and `decompress_string()` in the `utils.py`  

## 0.66.2 (Oct 15, 2025)
- added `serialize()` and `deserialize()` to `State`, `StateEnsemble`, and `StateNetwork` classes
- added `get_population()`, `get_micro_pKa()`, `get_macro_pKa()`, `get_pH_population_chart()` to `StateEnsemble`  class.
- made `StateNetwork` class call the `StateEnsemble` class methods for above functions.
- added `trim()` to `StateEnsemble` and `StateNetwork` classes
  
## 0.65.2 (Oct 12, 2025)
- added `Boltzmann_weights()`, `beta_constant()` at `microstate.py`
- revised `Boltzmann_weighted_average()` to take list or np.ndarray

## 0.64.2 (Oct 10, 2025)
- renamed functions and added @property decorators for `Conf` and `Mol`
- splitted `test_basic.py` into `test_conf.py`, `test_mol.py`, and `test_mollibr.py`
- revised serialize and deserialize functions for `Conf` and `Mol`
- cleaned up batch related functions
  
## 0.63.3 (Oct 8, 2025)
- added `networkx` dependency
  
## 0.63.2 (Oct 5, 2025)
- added `notebook/` documentation 
- added `State.energy`

## 0.62.1 (Oct 3, 2025)
- removed `mendeleev` dependency; replaced by rdkit.Chem.GetPeriodicTable()
- added `altair` dependency for `StateNetwork.population_chart()`
   
## 0.61.5 (Oct 3, 2025)
- added `State`, `StateEnsemble`, and `StateNetwork` classes to `microstate.py`
  
## 0.60.1 (Sep 6, 2025)
- added `to_molblock()` in `Mol`
- `Mol.serialize()` contains `rdmol`
  
## 0.59.3 (Sep 5, 2025)
- changed to default `compressed=True` in `serialize()` and `deserialize()` functions.

## 0.59.2 (Aug 30, 2025)
- fixed calculator(batch.rdmols, **kwargs).run() instead of calculator(batch.rdmols).run(**kwargs)

## 0.59.1 (Aug 28, 2025)
- added `prep_confs_batches()` in `conf.py`
- added `batch_optimize()` in `conf.py`
- added `batch_singlepoint()` in `conf.py`
- added `Mol.singlepoint()` that runs `batch_singlepoint()` if `batchsize_atoms` is set.
- renamed `Conf.potential_energy()` to `Conf.singlepoint()`

## 0.58.1 (Aug 27, 2025)
- added `Conf.torsion_energies_batch()` to use BatchOptimze
  
## 0.57.1 (July 24, 2025)
- reorient R-group such that attachment vector between scaffold and R-group align
- `MatchedSeries.report()` writes HTML files.
  
## 0.56.1 (July 23, 2025)
- maintain stereochemistry during tautomer enumeration
   
## 0.55.1 (July 23, 2025)
- put `complete_stereoisomers()` and `complete_tautomers()` in `complete.py`
- debugged `conf.py`:` data['angle'].append(conf.props['angle'])`
  
## 0.54.1 (July 11, 2025)
- removed `pka.py`
- changed APIs of `MicroStates()` class
- added `encoding='utf-8'` to the subprocess in the xtb wrapper

## 0.53.1 (July 2, 2025)
- added `count_stereoisomers()` to `Mol()` class
- moved `complete_stereoisomers()` from `stereoisomers.py` to `workflow.py`
- moved `complete_tautomers()` from `tautomers.py` to `workflow.py`
- 
## 0.52.1
- added `MicroStates()` class
- added `-c` charge option to the `GFNxTB()` subprocess.run()
- added `charge()` method to the `Conf()` class
- removed `cpx()` method from the `GFNxTB()` class
- added `tempfile` to the file I/O tests
  
## 0.51.1
- added `protonate()` and `deprotonate()` to `Conf()` class
- added `from_molblock()` to `Mol()` class
  
## 0.50.1
- added `optimize` test in `GFN2xTB()` class

## 0.49.1
- added `calculator` as a `**kwargs` key in some functions including `drop_confs()`
  
## 0.48.1
- revised `IonizedStates()` class for the pKa prediction
- reorganized `tests/`
  
## 0.47.1
- added `**kwargs` option to `sort_confs()` and `optimize_confs()` in `Mol()` class.
- fixed `GFN2xTB().singlepoint()` output handling in the `Conf().potential_energy()`

## 0.44.1
- added `to_turbomole_coord()` to `Conf()` and `GFN2xTB()` classes that generates strings of the TurboMole `coord` file.
- added `is_cpcmx_ready()` to `GFN2xTB()` class to check `--cpcmx` option is available in xtb.
added `is_cpx_ready()` to `GFN2xTB()` class to check `cpx` is accessible in the system.
- added the `CPCM-X` method to `GFN2xTB()` class.

## 0.43.2
- fixed processing of `GFN2xTB().optimize()` return value (SimpleNamespace)
  
## 0.43.1
- added `to_smi()` to `Conf()`. SMILES containing explicit hydrogens.

## 0.42.1
- added boolean outputs (`frag_created` and `wbo_filtered`) to `torsion.create_torsion_fragment()`: 
    `(frag_rdmol, frag_indices, frag_created, wbo_filtered) = create_torsion_fragment(rdmol, indices)`

## 0.41.1
- added `compressed` option to `Conf.__init__()` for molblock string

## 0.40.2
- fixed `Conf.__init__()` to initialize with molblock string
- `Chem.MolFromMolBlock(sanitize=False, removeHs=False, strictParsing=True)`

## 0.40.1
- added a dictionary `drugs` in `testdata.py`
- added `compress_string()` and `decompress_string()` to `utils.py`
- added `compressed(default: False)` argument to `serialize()` and `deserialize()` of `Conf` and `Mol`
- added `compressed(default: False)` argument to `Conf.to_molblock()`
  
## 0.39.1
- added `torsion_energies_one()` to `Conf`
- fixed `serialize()` and `deserialize()`
  
## 0.38.1
- added `generate_inchi_key()` to `rdworks.std`
  
## 0.37.1
- added `torsion_energies()` to `Conf` (and use it in `Mol`)
- added `torsion_atoms()` to `Conf`
- added rdDepictor.Compute2DCoords(mol) to `std.clean_2d()`

## 0.36.4
- fixed `create_torsion_fragment()` for small molecule that does not need fragmentation
- fixed `CONFORGE` tempfile issue
  
## 0.28
- added an option to torsion profile that uses a fragment instead of the full molecule.
- adopted `mendeleev` package to handle periodic table elements such as radii

## 0.27
- added xtb.wrapper singlepoint and optimize
- migrate from including external pytorch dependent pacakges: aimnet2 and auto3d
- support AIMNet2 through ASE calculator
- migrate from ANI-2x to AIMNet2
- renamed .is_nn_applicable() as .nnp_ready() with default model
- changed `decimal_places` to `decimals`
- changed conformer generation methods `RDKit_ETKDG` to `ETKDG` and `CDPL_CONFORGE` to `CONFORGE`
- renamed .serialize() as .dumps()
- added MolBlock interface for Conf: Conf(<molblockstring>) and Conf.to_molblock()
- added MolLibr.align_drawing()
- added Mol.rotate_drawing()
