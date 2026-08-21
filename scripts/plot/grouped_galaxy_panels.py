#!/usr/bin/env python3

"""Render all galaxy diagnostics as vector-native, physically grouped PDFs."""

import argparse
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas

from paper_galaxies import GALAXY_GROUPS, GALAXY_LABELS, PAPER_GALAXIES

MODEL_PLOTS = OrderedDict(
    (
        ("v_c.pdf", "circular velocity v_c"),
        ("v_R.pdf", "radial velocity v_R"),
        ("surface_rates.pdf", "surface rates"),
        ("cumulative_landing_rate.pdf", "cumulative landing rate"),
        ("Mdot_acc.pdf", "accretion rate"),
        ("t.pdf", "timescales"),
        ("Z.pdf", "metallicity"),
        ("mu.pdf", "mixing fraction mu"),
    )
)

MODEL_FAMILY_LABELS = {
    "forward": "forward model",
    "inverse": "inverse-dynamics model",
}


def all_galaxies():
    return PAPER_GALAXIES


def validate_groups():
    galaxies = all_galaxies()
    if len(galaxies) != len(set(galaxies)):
        raise ValueError("A galaxy appears in more than one physical group")
    if set(galaxies) != set(GALAXY_LABELS):
        raise ValueError("GALAXY_LABELS and GALAXY_GROUPS do not match")


def plot_command():
    configured = os.environ.get("GNF_PLOT_PYTHON")
    if configured:
        return [configured]
    if importlib.util.find_spec("matplotlib") and importlib.util.find_spec(
        "smplotlib"
    ):
        return [sys.executable]
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "No plotting Python was found. Set GNF_PLOT_PYTHON to a Python "
            "with matplotlib, smplotlib, and the project plotting dependencies."
        )
    return [conda, "run", "-n", "base", "python"]


def run_plot(command, script, arguments, repository_root, matplotlib_cache):
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    subprocess.run(
        [*command, str(script), *map(str, arguments)],
        cwd=repository_root,
        env=environment,
        check=True,
    )


def render_temporary_plots(repository_root, temporary_root):
    command = plot_command()
    model_script = repository_root / "scripts" / "plot" / "model_profiles.py"
    comparison_script = (
        repository_root / "scripts" / "plot" / "metallicity_comparison.py"
    )
    matplotlib_cache = temporary_root / "matplotlib"
    matplotlib_cache.mkdir()

    for family in MODEL_FAMILY_LABELS:
        for galaxy in all_galaxies():
            destination = temporary_root / "model" / family / galaxy
            destination.mkdir(parents=True)
            print(f"Rendering temporary model panels: {family}/{galaxy}", flush=True)
            run_plot(
                command,
                model_script,
                (
                    repository_root / "output" / family / galaxy,
                    destination,
                    "--no-title",
                ),
                repository_root,
                matplotlib_cache,
            )

    comparison_destination = temporary_root / "comparison"
    comparison_destination.mkdir()
    print("Rendering temporary SINGS/KK04 comparison panels", flush=True)
    run_plot(
        command,
        comparison_script,
        (
            repository_root / "output" / "comparison",
            comparison_destination,
            "--no-title",
        ),
        repository_root,
        matplotlib_cache,
    )


def read_single_page(path):
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected one page in {path}, found {len(reader.pages)}")
    return reader.pages[0]


def add_text_overlay(page, page_width, page_height, slots, missing):
    stream = BytesIO()
    overlay = canvas.Canvas(stream, pagesize=(page_width, page_height))

    for slot in slots:
        overlay.setFont("Helvetica-Bold", 11)
        overlay.drawCentredString(slot["center_x"], slot["label_y"], slot["label"])
        if slot["galaxy"] in missing:
            overlay.setFillColorRGB(0.35, 0.35, 0.35)
            overlay.setFont("Helvetica-Oblique", 10)
            overlay.drawCentredString(
                slot["center_x"],
                slot["panel_y"] + slot["panel_height"] / 2.0,
                "No SINGS/KK04 comparison data",
            )
            overlay.setFillColorRGB(0.0, 0.0, 0.0)

    overlay.save()
    stream.seek(0)
    page.merge_page(PdfReader(stream).pages[0])


def compose_group(
    paths,
    galaxies,
    title,
    destination,
    scale,
    layout_template=None,
    column_count=None,
):
    available_pages = {
        galaxy: read_single_page(path)
        for galaxy, path in paths.items()
        if path is not None and path.is_file()
    }
    layout_pages = list(available_pages.values())
    if not layout_pages and layout_template is not None:
        layout_pages.append(read_single_page(layout_template))
    if not layout_pages:
        raise ValueError(f"No source panels are available for {title}")

    source_width = max(
        float(source_page.mediabox.width)
        for source_page in layout_pages
    )
    source_height = max(
        float(source_page.mediabox.height)
        for source_page in layout_pages
    )
    columns = (
        column_count
        if column_count is not None
        else (2 if len(galaxies) == 4 else min(3, len(galaxies)))
    )
    if columns < 1:
        raise ValueError("Grouped panels require at least one column")
    rows = math.ceil(len(galaxies) / columns)
    margin_x = 18.0
    margin_bottom = 18.0
    header_height = 12.0
    label_height = 18.0
    gap_x = 10.0
    gap_y = 10.0
    panel_width = source_width * scale
    panel_height = source_height * scale
    cell_height = panel_height + label_height
    content_width = (
        2.0 * margin_x + columns * panel_width + (columns - 1) * gap_x
    )
    page_width = content_width
    content_offset = 0.5 * (page_width - content_width)
    page_height = (
        margin_bottom
        + header_height
        + rows * cell_height
        + (rows - 1) * gap_y
    )
    output_page = PageObject.create_blank_page(width=page_width, height=page_height)
    slots = []

    for index, galaxy in enumerate(galaxies):
        row = index // columns
        column = index % columns
        panel_x = content_offset + margin_x + column * (panel_width + gap_x)
        cell_top = page_height - header_height - row * (cell_height + gap_y)
        panel_y = cell_top - label_height - panel_height
        source_page = available_pages.get(galaxy)
        if source_page is not None:
            width = float(source_page.mediabox.width)
            height = float(source_page.mediabox.height)
            source_x = panel_x + 0.5 * (source_width - width) * scale
            source_y = panel_y + 0.5 * (source_height - height) * scale
            transform = Transformation().scale(scale).translate(source_x, source_y)
            output_page.merge_transformed_page(source_page, transform)
        slots.append(
            {
                "galaxy": galaxy,
                "label": GALAXY_LABELS[galaxy],
                "center_x": panel_x + panel_width / 2.0,
                "label_y": cell_top - 12.0,
                "panel_y": panel_y,
                "panel_height": panel_height,
            }
        )

    add_text_overlay(
        output_page,
        page_width,
        page_height,
        slots,
        set(galaxies) - set(available_pages),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(output_page)
    writer.add_metadata(
        {
            "/Title": title,
            "/Subject": "Physically grouped Galactic Nuclear Fountain diagnostics",
        }
    )
    with destination.open("wb") as stream:
        writer.write(stream)


def compose_all(repository_root, temporary_root, output_root):
    expected = set()
    for family, family_label in MODEL_FAMILY_LABELS.items():
        for group_slug, (group_title, galaxies) in GALAXY_GROUPS.items():
            for filename, diagnostic_label in MODEL_PLOTS.items():
                destination = output_root / "model" / family / group_slug / filename
                paths = {
                    galaxy: temporary_root / "model" / family / galaxy / filename
                    for galaxy in galaxies
                }
                compose_group(
                    paths,
                    galaxies,
                    f"{group_title} - {family_label} - {diagnostic_label}",
                    destination,
                    scale=1.0,
                )
                expected.add(destination.resolve())

    for family, family_label in MODEL_FAMILY_LABELS.items():
        layout_template = next(
            (
                temporary_root
                / "comparison"
                / galaxy
                / f"{family}_KK04_comparison.pdf"
                for galaxy in all_galaxies()
                if (
                    temporary_root
                    / "comparison"
                    / galaxy
                    / f"{family}_KK04_comparison.pdf"
                ).is_file()
            ),
            None,
        )
        for group_slug, (group_title, galaxies) in GALAXY_GROUPS.items():
            destination = output_root / "comparison" / family / f"{group_slug}.pdf"
            paths = {}
            for galaxy in galaxies:
                path = (
                    temporary_root
                    / "comparison"
                    / galaxy
                    / f"{family}_KK04_comparison.pdf"
                )
                paths[galaxy] = path if path.is_file() else None
            compose_group(
                paths,
                galaxies,
                f"{group_title} - {family_label} - SINGS/KK04 comparison",
                destination,
                scale=0.85,
                layout_template=layout_template,
                column_count=1,
            )
            expected.add(destination.resolve())

    for stale in output_root.rglob("*.pdf"):
        if stale.resolve() not in expected:
            stale.unlink()
    for directory in sorted(
        (path for path in output_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    expected_count = (
        len(MODEL_FAMILY_LABELS) * len(GALAXY_GROUPS) * (len(MODEL_PLOTS) + 1)
    )
    if len(expected) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} grouped PDFs, assembled {len(expected)}"
        )
    return expected


def remove_individual_pdfs(repository_root):
    removed = []
    for family in MODEL_FAMILY_LABELS:
        for path in (repository_root / "output" / family).glob("*/plots/*.pdf"):
            path.unlink()
            removed.append(path)
        for directory in (repository_root / "output" / family).glob("*/plots"):
            try:
                directory.rmdir()
            except OSError:
                pass
    for path in (repository_root / "output" / "comparison").glob("*/*.pdf"):
        path.unlink()
        removed.append(path)
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate vector-native galaxy plots in physically grouped panels"
    )
    repository_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
        help="repository root; inferred from this script by default",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="grouped output directory; defaults to REPOSITORY_ROOT/output/grouped",
    )
    parser.add_argument(
        "--clean-individual",
        action="store_true",
        help="remove legacy per-galaxy PDFs after grouped PDFs are assembled",
    )
    arguments = parser.parse_args()
    repository_root = arguments.repository_root.resolve()
    output_root = (
        arguments.output_root.resolve()
        if arguments.output_root
        else repository_root / "output" / "grouped"
    )

    validate_groups()
    with tempfile.TemporaryDirectory(prefix="gnf-grouped-panels-") as temporary:
        temporary_root = Path(temporary)
        render_temporary_plots(repository_root, temporary_root)
        expected = compose_all(repository_root, temporary_root, output_root)

    removed = []
    if arguments.clean_individual:
        removed = remove_individual_pdfs(repository_root)
    print(f"Wrote {len(expected)} grouped PDFs to: {output_root}")
    if arguments.clean_individual:
        print(f"Removed {len(removed)} legacy individual PDFs")


if __name__ == "__main__":
    main()
