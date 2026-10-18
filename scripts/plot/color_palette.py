"""Shared categorical colors: Petroff's rank-3 ten-color set.

Exact hex values from the Ten Colors column, rank 3, in the published
result-plots.ipynb table:
https://github.com/mpetroff/accessible-color-cycles/blob/master/other-analysis/result-plots.ipynb

The set was selected with color-vision-deficiency distance constraints. Ten
colors do not distinguish an unlimited number of curves: repeated colors get
an additional line style, while physical quantity styles stay with the caller.
"""

from cycler import cycler


PETROFF_RANK3 = (
    "#9fa7ad", "#94c4f4", "#4c8af2", "#6e32f7", "#9d6c9f",
    "#7d426c", "#bf241a", "#df6a09", "#edc22d", "#587d63",
)
# Use well-separated, darker colors first for smaller subsets on white paper.
COLOR_CYCLE = tuple(PETROFF_RANK3[i] for i in (2, 6, 9, 3, 7, 5, 1, 4, 8, 0))
FIDUCIAL_COLOR = "black"
REFERENCE_COLOR = PETROFF_RANK3[0]
BLUE = PETROFF_RANK3[2]
ORANGE = PETROFF_RANK3[7]
REFERENCE_CASES = {"unmixed", "unmixed_fiducial"}


def color_cycle():
    """Matplotlib rcParams value for plots without explicit case colors."""
    return cycler(color=COLOR_CYCLE)


def apply_case_palette(rows):
    """Copy a case table, assigning palette colors and keeping its case order.

Black identifies the fiducial. An unmixed reference uses the palette's gray;
that gray is then excluded from the parameter colors. Existing special line
styles are retained. After exhausting the colors, dash-dot strokes distinguish
the second pass (dotted strokes distinguish the third).
"""
    rows = [dict(row) for row in rows]
    reference_present = any(row["case"] in REFERENCE_CASES for row in rows)
    colors = tuple(c for c in COLOR_CYCLE if not reference_present or c != REFERENCE_COLOR)
    index = 0
    for row in rows:
        if row["case"] == "fiducial":
            row["color"] = FIDUCIAL_COLOR
        elif row["case"] in REFERENCE_CASES:
            row["color"] = REFERENCE_COLOR
        else:
            row["color"] = colors[index % len(colors)]
            if index >= len(colors) and row.get("linestyle", "-") in ("", "-"):
                row["linestyle"] = "-." if index < 2 * len(colors) else ":"
            index += 1
    return rows


def palette_case_styles(styles):
    """Apply the same convention to case -> (color, linestyle, label) maps."""
    rows = apply_case_palette([
        dict(case=name, color=color, linestyle=linestyle, label=label)
        for name, (color, linestyle, label) in styles.items()
    ])
    return {row["case"]: (row["color"], row.get("linestyle") or "-", row["label"]) for row in rows}
