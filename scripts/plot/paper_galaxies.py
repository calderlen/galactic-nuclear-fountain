"""Canonical galaxy selection and labels for all paper plots."""

from collections import OrderedDict


GALAXY_GROUPS = OrderedDict(
    (
        (
            "gas_rich_dwarf_irregulars",
            ("Gas-rich dwarf irregular", ("UGC4305",)),
        ),
        (
            "lower_intermediate_late_type_disks",
            (
                "Lower/intermediate-mass late-type disks",
                ("NGC7793", "NGC2403", "NGC3198"),
            ),
        ),
        (
            "massive_spirals",
            ("Massive spirals", ("NGC6946", "NGC5055", "NGC3521")),
        ),
    )
)

GALAXY_LABELS = {
    "UGC4305": "Ho II",
    "NGC7793": "NGC 7793",
    "NGC2403": "NGC 2403",
    "NGC3198": "NGC 3198",
    "NGC6946": "NGC 6946",
    "NGC5055": "M63",
    "NGC3521": "NGC 3521",
}

PAPER_GALAXIES = tuple(
    galaxy for _, galaxies in GALAXY_GROUPS.values() for galaxy in galaxies
)
PAPER_GALAXY_SET = frozenset(PAPER_GALAXIES)

_ALIASES = {
    "HOII": "UGC4305",
    "HOLMBERGII": "UGC4305",
    "M63": "NGC5055",
}


def canonical_galaxy_name(name):
    compact = "".join(character for character in name.upper() if character.isalnum())
    return _ALIASES.get(compact, compact)


def galaxy_label(galaxy):
    return GALAXY_LABELS.get(galaxy, galaxy)


def require_paper_galaxy(galaxy):
    if galaxy not in PAPER_GALAXY_SET:
        allowed = ", ".join(GALAXY_LABELS[item] for item in PAPER_GALAXIES)
        raise ValueError(
            f"Plotting is restricted to the paper sample ({allowed}); got {galaxy}"
        )
