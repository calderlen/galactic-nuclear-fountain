"""Keep rotated y-axis quantities and units on a single line."""

import re


def quantity_unit_label(label):
    """Join explicit line breaks while preserving quantity, unit and math."""
    if not isinstance(label, str):
        return label
    return re.sub(r"\s*\n\s*", " ", label)
