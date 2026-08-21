#!/usr/bin/env python3

"""Extract the published Bigiel et al. (2010) radial points from Figure 2.

The paper's electronic supplement contains pixel-distribution data, but not
the galaxy-by-galaxy radial profiles plotted in Figure 2.  The arXiv source
does contain vector EPS versions of that figure.  This one-time extractor
recovers the plotted marker centers and vertical error bars for the galaxies
used by this project, retaining only the outer-disk measurements at R/R25 >= 1.
"""

import argparse
import csv
import math
import re
from pathlib import Path


AXIS_X = {
    "left": (2442.0, 8939.0),
    "right": (12047.0, 18544.0),
}
AXIS_Y = {
    "top": (17872.0, 23992.0),
    "middle": (9640.0, 15760.0),
    "bottom": (1408.0, 7528.0),
}

# (file, column, row).  These are the seven galaxies shared by the current
# Q=1 SPARC/Leroy/THINGS sample and the Bigiel et al. (2010) sample.
PANELS = {
    "NGC2403": ("figure2-1.eps", "right", "bottom"),
    "NGC2841": ("figure2-2.eps", "left", "top"),
    "NGC3198": ("figure2-2.eps", "left", "middle"),
    "NGC3521": ("figure2-2.eps", "left", "bottom"),
    "NGC5055": ("figure2-3.eps", "left", "middle"),
    "NGC7331": ("figure2-4.eps", "right", "top"),
    "NGC7793": ("figure2-4.eps", "left", "middle"),
}

EXPECTED_OUTER_POINT_COUNTS = {
    "NGC2403": 31,
    "NGC2841": 14,
    "NGC3198": 13,
    "NGC3521": 16,
    "NGC5055": 24,
    "NGC7331": 19,
    "NGC7793": 21,
}

TOKEN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
    r"|(?<![A-Za-z])(?:setrgbcolor|setlinewidth|[MPRFDK])(?![A-Za-z])"
)


def color_name(rgb):
    if max(abs(component) for component in rgb) < 0.02:
        return "HI"
    if max(abs(component - 0.647) for component in rgb) < 0.02:
        return "SFR"
    return None


def parse_vector_paths(path):
    text = path.read_text(encoding="ascii", errors="ignore")
    page = text.split("%%BeginPageSetup", maxsplit=1)[-1]
    numbers = []
    current = None
    current_path = []
    rgb = (0.0, 0.0, 0.0)
    line_width = math.nan
    filled_paths = []
    stroked_paths = []

    for token in TOKEN.findall(page):
        try:
            numbers.append(float(token))
            continue
        except ValueError:
            pass

        if token == "setrgbcolor":
            rgb = tuple(numbers[-3:])
            del numbers[-3:]
        elif token == "K":
            gray = numbers.pop()
            rgb = (gray, gray, gray)
        elif token == "setlinewidth":
            line_width = numbers.pop()
        elif token == "M":
            x, y = numbers[-2:]
            del numbers[-2:]
            current = (x, y)
            current_path = [current]
        elif token == "R" and current is not None:
            dx, dy = numbers[-2:]
            del numbers[-2:]
            current = (current[0] + dx, current[1] + dy)
            current_path.append(current)
        elif token == "P" and current is not None:
            x, y = numbers[-2:]
            del numbers[-2:]
            current = (x, y)
            current_path.append(current)
        elif token == "F":
            if current_path:
                filled_paths.append((rgb, line_width, tuple(current_path)))
            current_path = []
        elif token == "D":
            if current_path:
                stroked_paths.append((rgb, line_width, tuple(current_path)))
            current_path = []

    return filled_paths, stroked_paths


def marker_center(points):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    width = max(x_values) - min(x_values)
    height = max(y_values) - min(y_values)
    if not (65.0 <= width <= 72.0 and 65.0 <= height <= 72.0):
        return None
    return (
        0.5 * (min(x_values) + max(x_values)),
        0.5 * (min(y_values) + max(y_values)),
    )


def in_panel(point, x_bounds, y_bounds):
    return (
        x_bounds[0] <= point[0] <= x_bounds[1]
        and y_bounds[0] <= point[1] <= y_bounds[1]
    )


def radial_coordinate(x, x_bounds):
    return 2.0 * (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])


def surface_density(y, y_bounds, lower_log10):
    log_value = lower_log10 + 4.0 * (
        (y - y_bounds[0]) / (y_bounds[1] - y_bounds[0])
    )
    return 10.0**log_value


def vertical_error(marker, color, stroked_paths, x_bounds, y_bounds, lower_log10):
    candidates = []
    for path_color, line_width, points in stroked_paths:
        if color_name(path_color) != color or line_width != 10.0 or len(points) != 2:
            continue
        (x0, y0), (x1, y1) = points
        if abs(x1 - x0) > 0.5 or abs(x0 - marker[0]) > 0.5:
            continue
        if not (x_bounds[0] <= x0 <= x_bounds[1]):
            continue
        lower_y, upper_y = sorted((y0, y1))
        if upper_y < y_bounds[0] or lower_y > y_bounds[1]:
            continue
        if lower_y <= marker[1] <= upper_y:
            candidates.append((upper_y - lower_y, lower_y, upper_y))

    if not candidates:
        raise ValueError(
            f"No {color} vertical error bar found at EPS coordinate {marker}"
        )

    _, lower_y, upper_y = min(candidates)
    lower_value = surface_density(lower_y, y_bounds, lower_log10)
    upper_value = surface_density(upper_y, y_bounds, lower_log10)
    return 0.5 * (upper_value - lower_value)


def extract_panel(filled_paths, stroked_paths, column, row):
    x_bounds = AXIS_X[column]
    y_bounds = AXIS_Y[row]
    markers = {"HI": [], "SFR": []}

    for rgb, _line_width, points in filled_paths:
        source = color_name(rgb)
        center = marker_center(points)
        if source is None or center is None or not in_panel(center, x_bounds, y_bounds):
            continue
        if radial_coordinate(center[0], x_bounds) >= 1.0 - 1.0e-6:
            markers[source].append(center)

    for source in markers:
        markers[source].sort()

    rows = []
    unmatched_sfr = list(markers["SFR"])
    for hi_marker in markers["HI"]:
        matches = [
            marker for marker in unmatched_sfr if abs(marker[0] - hi_marker[0]) <= 1.0
        ]
        if not matches:
            continue
        sfr_marker = min(matches, key=lambda marker: abs(marker[0] - hi_marker[0]))
        unmatched_sfr.remove(sfr_marker)

        sigma_hi = surface_density(hi_marker[1], y_bounds, -2.0)
        sigma_sfr = surface_density(sfr_marker[1], y_bounds, -6.0)
        e_sigma_hi = vertical_error(
            hi_marker, "HI", stroked_paths, x_bounds, y_bounds, -2.0
        )
        e_sigma_sfr_stat = vertical_error(
            sfr_marker, "SFR", stroked_paths, x_bounds, y_bounds, -6.0
        )
        rows.append(
            {
                "R_R25": radial_coordinate(hi_marker[0], x_bounds),
                "SigmaHI_Msun_pc2": sigma_hi,
                "e_SigmaHI_Msun_pc2": e_sigma_hi,
                "SigmaSFR_Msun_yr_kpc2": sigma_sfr,
                "e_SigmaSFR_stat_Msun_yr_kpc2": e_sigma_sfr_stat,
                # Bigiel et al. adopt an approximately 50% uncertainty in
                # converting FUV intensity to SFR surface density.
                "e_SigmaSFR_Msun_yr_kpc2": math.hypot(
                    e_sigma_sfr_stat, 0.5 * sigma_sfr
                ),
            }
        )

    return rows


def extract(source_directory):
    parsed_files = {}
    output_rows = []
    for galaxy, (filename, column, row) in PANELS.items():
        if filename not in parsed_files:
            parsed_files[filename] = parse_vector_paths(source_directory / filename)
        galaxy_rows = extract_panel(*parsed_files[filename], column, row)
        expected = EXPECTED_OUTER_POINT_COUNTS[galaxy]
        if len(galaxy_rows) != expected:
            raise ValueError(
                f"Extracted {len(galaxy_rows)} outer points for {galaxy}; "
                f"expected {expected}"
            )
        for row_values in galaxy_rows:
            output_rows.append({"galaxy": galaxy, **row_values})
    return output_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output_csv", type=Path)
    arguments = parser.parse_args()

    rows = extract(arguments.source_directory)
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "galaxy",
        "R_R25",
        "SigmaHI_Msun_pc2",
        "e_SigmaHI_Msun_pc2",
        "SigmaSFR_Msun_yr_kpc2",
        "e_SigmaSFR_stat_Msun_yr_kpc2",
        "e_SigmaSFR_Msun_yr_kpc2",
    ]
    with arguments.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Bigiel et al. (2010) radial points to {arguments.output_csv}")


if __name__ == "__main__":
    main()
