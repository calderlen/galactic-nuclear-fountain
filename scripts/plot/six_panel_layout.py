"""Compact six-panel figures without changing their physical panel sizes."""


def _text_boxes(axis, renderer):
    artists = [axis.xaxis.label, axis.yaxis.label, axis.xaxis.offsetText,
               axis.yaxis.offsetText, axis.title, *axis.texts]
    for coordinate in (axis.xaxis, axis.yaxis):
        # Locators can include ticks outside the view. Only drawn ticks count.
        for tick in coordinate._update_ticks():
            artists.extend((tick.label1, tick.label2))
    legend = axis.get_legend()
    if legend is not None:
        artists.extend(legend.get_texts())
    boxes = [text.get_window_extent(renderer) for text in artists if text.get_visible() and text.get_text().strip()]
    for child in axis.child_axes:
        boxes.extend(_text_boxes(child, renderer))
    return boxes


def compact_six_panel_rows(figure, axes, clearance_pt=4):
    """Remove surplus row gaps, retaining clearance between neighboring text.

    Compare text only where its horizontal extents overlap. Whole-axis bounding
    rectangles otherwise reserve unnecessary space around rotated y labels.
    Keep the bottom row and shared legend fixed; shift the other rows and title
    downward, then remove the freed page height from the top.
    """
    axes = list(axes)
    if len(axes) != 6:
        raise ValueError("Expected a three-row, two-column figure")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    positions = [axis.bbox.frozen() for axis in axes]
    texts = [_text_boxes(axis, renderer) for axis in axes]
    clearance = clearance_pt * figure.dpi / 72
    savings = []
    for row in range(2):
        required = clearance  # Keep the panel borders apart, too.
        for column in range(2):
            upper, lower = 2 * row + column, 2 * row + column + 2
            for a in texts[upper]:
                for b in texts[lower]:
                    if a.x0 < b.x1 and b.x0 < a.x1:
                        required = max(required, clearance + b.y1 - positions[lower].y1 + positions[upper].y0 - a.y0)
        existing = positions[2 * row].y0 - positions[2 * row + 2].y1
        savings.append(max(existing - required, 0))
    removed = sum(savings)
    if removed == 0:
        return
    width, height = figure.bbox.width, figure.bbox.height
    anchors = [legend.get_bbox_to_anchor().frozen() for legend in figure.legends]
    heading = figure._suptitle
    heading_position = heading.get_position() if heading is not None else None
    figure.set_size_inches(width / figure.dpi, (height - removed) / figure.dpi)
    new_height = figure.bbox.height
    for index, (axis, box) in enumerate(zip(axes, positions)):
        shift = sum(savings[index // 2:])
        axis.set_position((box.x0 / width, (box.y0 - shift) / new_height, box.width / width, box.height / new_height))
    for legend, box in zip(figure.legends, anchors):
        legend.set_bbox_to_anchor((box.x0 / width, box.y0 / new_height, box.width / width, box.height / new_height),
                                  transform=figure.transFigure)
    if heading_position is not None:
        heading.set_position((heading_position[0], (heading_position[1] * height - removed) / new_height))
