def test_version():
    from rdworks import __version__, __rdkit_version__
    import rdkit
    import logging

    logger = logging.getLogger(__name__)
    
    assert isinstance(__version__, str)
    assert isinstance(__rdkit_version__, str)
    assert __rdkit_version__ == rdkit.rdBase.rdkitVersion

    logger.info(f"rdworks version: {__version__}")
    logger.info(f"RDKit version: {__rdkit_version__}")