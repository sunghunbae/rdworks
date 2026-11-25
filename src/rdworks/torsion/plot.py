def get_torsion_plot_data(torsion_profiles: list[dict], idx: int | None = None) -> dict:
    """Returns plot data (torsion angle vs. relative energy) from torsion scan results.

    Args:
        torsion_profiles (list[dict]): torsion profiles, i.e. `props['torsion']` set by
            `calculate_torsion_energies()` or `calculate_sp_torsion_energies()`.
        idx (int, optional): 0-based torsion angle index.
            Defaults to None (all torsion angles).

    Returns:
        {'idx': list, 'angle': list, 'E_rel(kcal/mol)': list}
    """
    if isinstance(idx, int):
        _dict = torsion_profiles[idx]
        angles, energies = _dict["angle"], _dict["E_rel(kcal/mol)"]
        assert len(angles) == len(energies)
        data = {
            "idx": [
                idx,
            ]
            * len(angles),
            "angle": angles,
            "E_rel(kcal/mol)": energies,
        }
    else:
        data = {"idx": [], "angle": [], "E_rel(kcal/mol)": []}
        for idx, _dict in enumerate(torsion_profiles):  # list
            angles, energies = _dict["angle"], _dict["E_rel(kcal/mol)"]
            assert len(angles) == len(energies)
            data["idx"].extend(
                [
                    idx,
                ]
                * len(angles)
            )
            data["angle"].extend(angles)
            data["E_rel(kcal/mol)"].extend(energies)

    return data