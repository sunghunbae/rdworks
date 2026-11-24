## Conformer Generation

To yield more chemically meaningful conformers, 
Riniker and Landrum implemented the experimental torsion knowledge distance geometry (ETKDG) method 
which uses torsion angle preferences from the Cambridge Structural Database (CSD) 
to correct the conformers after distance geometry has been used to generate them. 

The configs of various conformer generation options are stored in a EmbedParameter object:

```python 
# To explicitly call the ETKDG EmbedParameter object: 
params = AllChem.ETKDG() 
params = AllChem.ETKDGv2()

# Addtional small ring torsion potentials: 
params = AllChem.srETKDGv3()

# Additional macrocycle ring torsion potentials and macrocycle-specific handles
params = AllChem.ETKDGv3()

# use the above two in conjunction, do:
params = AllChem.ETKDGv3()
params.useSmallRingTorsions = True

# default rdDistGeom.EmbedMultipleConfs() settings
    randomSeed=-1           RNG will not be seeded
    useRandomCoords=False
    pruneRmsThresh=-1.0     No pruning by default
    numThreads=1
    useRandomCoords=False   Start the embedding from random coordinates 
                            instead of using eigenvalues of the distance matrix
    numThreads=1            Number of threads to use while embedding
                            Only if built with multi-thread support
                            =0, the max supported by the system
```


### Number of conformers

| RotBonds | N_confs | RotBonds | N_confs |
| -------- | ------- | -------- | ------- |
| 0	       |    1    |    11	|     434 |
| 1	       |    8    |    12	|     501 |
| 2	       |   26    |    13	|     572 |
| 3	       |   51    |    14	|     646 |
| 4	       |   82    |    15	|     723 |
| 5	       |  119    |    16	|     804 |
| 6	       |  160    |    17	|     888 |
| 7	       |  207    |    18	|     976 |
| 8	       |  257    |    19	|    1000 |
| 9	       |  312    |    20	|    1000 |
|10	       |  371    |          |         |

### ETKDG method 

The ETKDG method of Riniker and Landrum uses torsion angle preferences 
from the Cambridge Structural Database (CSD) to correct the conformers 
after distance geometry has been used to generate them. 
With this method, there should be no need to use a minimisation step 
to clean up the structures. Since the 2018.09 release of the RDKit, 
ETKDG is the default conformer generation method.

Optimization on a molecule’s conformers is not often necessary when using ETKDG:
```python
res = AllChem.MMFFOptimizeMoleculeConfs(m)
```

Rms between conformers
```python
AllChem.AlignMolConformers(m, RMSlist=rmslist)
rms = AllChem.GetConformerRMS(m, 1, 9, prealigned=True)
```

Contrary to the above Greg's comment, ``RDKit ETKDG`` conformer quality actually improves
greatly by MMFF94 or MMFF94s optimization after embedding. UFF optimization is less
effective than MMFF94 or MMFF94s.

``Torsional Diffusion`` is the most efficient and accurate AI method according to Wang et al. (2023)

``Auto3D`` uses RDKit conformers as a starting point for optimizing with a modified version of the 
ANI-2x deep learning molecular potential (AIMNET), but this does not result in better performance 
than that of RDKit in the bioactive conformation identification task.

Generating a larger ensemble will monotonically increase the likelihood of  retrieving a  bioactive conformation, 
but for efficiency reasons, it is desirable to generate smaller ensembles. As shown in Figures 5 and S6,   
selecting the lowest energy poses from a larger ensemble does not improve the retrieval of bioactive conformations 
with the exception of reducing down to a single conformer.
        
### References
  - L. Chan, G. M. Morris, G. R. Hutchison, Understanding Conformational Entropy in Small Molecules. 
      J. Chem. Theory Comput. 17, 2099–2106 (2021).
  - Riniker, S.; Landrum, G. A. “Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation” 
      J. Chem. Inf. Comp. Sci. 55:2562-74 (2015)
  - Z. Wang, et al., Small-Molecule Conformer Generators: 
      Evaluation of Traditional Methods and AI Models on High-Quality Data Sets. 
      J. Chem. Inf. Model. 63, 6525–6536 (2023).
  - B. Jing, G. Corso, J. Chang, R. Barzilay, T. Jaakkola, 
      Torsional Diffusion for Molecular Conformer Generation. arXiv [physics.chem-ph] (2022).
  - A. T. McNutt, et al., Conformer Generation for Structure-Based Drug Design: How Many and How Good? 
    J. Chem. Inf. Model. 63, 6598–6607 (2023).


## Clustering: 

[Clustering conformers](https://greglandrum.github.io/rdkit-blog/posts/2023-03-02-clustering-conformers.html)

### Usage of `Butina.ClusterData()`

  - isDistData: set this toggle when the data passed in is a
  - distance matrix. The distance matrix should be stored symmetrically.
  - distFunc: a function to calculate distances between points.
      Receives 2 points as arguments, should return a float  
  - reordering: if this toggle is set, the number of neighbors is updated
      for the unassigned molecules after a new cluster is created 
      such that always the molecule with the largest number of unassigned neighbors 
      is selected as the next cluster center.
  - returns a tuple of tuples containing information about the clusters:
      ( 
          (cluster1_elem1, cluster1_elem2, …), 
          (cluster2_elem1, cluster2_elem2, …), 
          …,
      ) where the first element for each cluster is its centroid.
  - https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html
        
### References

- D. Butina, Unsupervised Data Base Clustering Based on Daylight’s Fingerprint and Tanimoto Similarity: 
    A Fast and Automated Way To Cluster Small and Large Data Sets. J. Chem. Inf. Comput. Sci. 39, 747–750 (1999).

## Fingerprint
[Tutorial](https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html)

```python
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

fpgen.GetFingerprint(m)               : returns a bit vector of size fpSize
fpgen.GetCountFingerprint(m)          : returns a count vector of size fpSize
fpgen.GetSparseFingerprint(m)         : returns a sparse bit vector
fpgen.GetSparseCountFingerprint(m)    : returns a sparse count vector
```
Note that sparse vector is very long.

## Similarity
[Similarity search thresholds](https://greglandrum.github.io/rdkit-blog/posts/2021-05-21-similarity-search-thresholds.html)

## Druglikeness and QED

### References
  - G. R. Bickerton, G. V. Paolini, J. Besnard, S. Muresan, A. L. Hopkins, 
    Quantifying the chemical beauty of drugs. Nat. Chem. 4, 90–98 (2012).

## Hydrogen Bonding Acceptor
[QED definition in RDkit](https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/QED.py)

```python
Acceptors = [
    Chem.MolFromSmarts(hba) for hba in [
        '[oH0;X2]', 
        '[OH1;X2;v2]', 
        '[OH0;X2;v2]', 
        '[OH0;X1;v2]', 
        '[O-;X1]', 
        '[SH0;X2;v2]', 
        '[SH0;X1;v2]',
        '[S-;X1]',
        '[nH0;X2]',
        '[NH0;X1;v3]',
        '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
    ]]
```

## API

::: rdworks.mol.Mol
    handler: python
    options:
        show_source: false