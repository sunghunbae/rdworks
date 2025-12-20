from importlib.resources import files, as_file
from rdworks.io import read_smi


def test_cluster():
    with as_file(files("rdworks.data.test") / "cdk2.smi.gz") as path:
        libr = read_smi(path, progress=False)
        assert libr.count() == 47
        clusters = libr.cluster(threshold=0.3)
        assert isinstance(clusters, list)
        assert len(clusters) == 3
