"""Canonical surface-rate CSV units: Msun/Gyr/kpc^2.

Imported archival/catalog rates keep their original files and are converted once
at the read boundary. A canonical Gyr column is never scaled again.
"""
import csv
from pathlib import Path

YEARS_PER_GYR = 1e9
MYR_PER_GYR = 1e3


def canonical_column(name):
    if "_Msun_yr_kpc2" in name:
        return name.replace("_Msun_yr_kpc2", "_Msun_Gyr_kpc2"), YEARS_PER_GYR
    if "_Msun_yr_Myr" in name:
        return name.replace("_Msun_yr_Myr", "_Msun_Gyr_Myr"), YEARS_PER_GYR
    if name.startswith("dSigma_") and name.endswith("_Msun_kpc2_Myr"):
        return name.replace("_Msun_kpc2_Myr", "_Msun_Gyr_kpc2"), MYR_PER_GYR
    return name, 1.0


def canonical_rate_row(row):
    result = {}
    for name, value in row.items():
        canonical, factor = canonical_column(name)
        if canonical in result:
            raise ValueError(f"Mixed/duplicate surface-rate units for {canonical}")
        if factor != 1 and value not in (None, "", "missing", "NA"):
            value = format(float(value)*factor, ".17g") if isinstance(value, str) else value*factor
        result[canonical] = value
    return result


def read_rate_rows(path):
    with Path(path).open(newline="") as stream:
        return [canonical_rate_row(row) for row in csv.DictReader(stream)]
