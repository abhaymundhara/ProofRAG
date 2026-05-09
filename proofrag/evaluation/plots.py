"""
plots.py — Dependency-free SVG chart helpers for evaluation reports.
"""

from __future__ import annotations

from html import escape
from pathlib import Path


def bar_chart_svg(
    values: dict[str, float],
    *,
    title: str,
    width: int = 720,
    height: int = 360,
) -> str:
    """Return a simple SVG bar chart for values in [0, 1]."""

    if width < 240 or height < 180:
        raise ValueError("chart dimensions are too small")

    margin_left = 140
    margin_right = 32
    margin_top = 52
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    bar_gap = 14
    count = max(1, len(values))
    bar_height = max(18, (height - margin_top - margin_bottom - bar_gap * (count - 1)) // count)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">{escape(title)}</text>',
    ]
    for index, (label, raw_value) in enumerate(values.items()):
        value = max(0.0, min(1.0, raw_value))
        y = margin_top + index * (bar_height + bar_gap)
        bar_width = plot_width * value
        lines.extend(
            [
                f'<text x="{margin_left - 12}" y="{y + bar_height * 0.65:.0f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13">{escape(label)}</text>',
                f'<rect x="{margin_left}" y="{y}" width="{plot_width}" height="{bar_height}" fill="#eef2f7"/>',
                f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="#2563eb"/>',
                f'<text x="{margin_left + bar_width + 8:.1f}" y="{y + bar_height * 0.65:.0f}" font-family="Arial, sans-serif" font-size="13">{value:.1%}</text>',
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines)


def write_bar_chart_svg(
    path: str | Path,
    values: dict[str, float],
    *,
    title: str,
) -> Path:
    """Write a bar chart SVG and return the path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bar_chart_svg(values, title=title), encoding="utf-8")
    return output_path

