"""Register smplotlib's bundled Hershey faces with Matplotlib weight aliases."""

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

from matplotlib import font_manager


def register_hershey_weight_aliases():
    """Use Hershey Medium for normal text and Hershey Heavy for bold text.

    smplotlib bundles light (200), medium (500), and heavy (800) Hershey
    faces, while Matplotlib requests normal (400) and bold (700) for ordinary
    text. Registering aliases makes those semantic requests select the bundled
    medium and heavy files directly instead of relying on nearest-weight
    fallback.
    """
    specification = find_spec("smplotlib")
    if specification is None or specification.submodule_search_locations is None:
        raise ImportError("smplotlib is required to register the Hershey fonts")

    package_directory = Path(next(iter(specification.submodule_search_locations)))
    for font_path in (package_directory / "ttf").glob("*.ttf"):
        font_manager.fontManager.addfont(font_path)

    registered = {
        (entry.fname, entry.style, entry.weight)
        for entry in font_manager.fontManager.ttflist
    }
    for entry in list(font_manager.fontManager.ttflist):
        if not entry.name.startswith("AVHershey"):
            continue

        alias_weight = {500: 400, 800: 700}.get(entry.weight)
        alias = (entry.fname, entry.style, alias_weight)
        if alias_weight is None or alias in registered:
            continue

        font_manager.fontManager.ttflist.append(
            replace(entry, weight=alias_weight)
        )
        registered.add(alias)
