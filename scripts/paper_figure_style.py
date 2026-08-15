"""Shared, dependency-free SVG styling for the paper figures.

The palette is based on the Okabe--Ito colour-blind-safe palette.  Important
comparisons also differ in marker shape or line style, so colour is never the
only carrier of meaning.
"""

from __future__ import annotations

from html import escape
from typing import Iterable


PALETTE = {
    "ink": "#1A1A1A",
    "muted": "#5C6670",
    "grid": "#D6DCE2",
    "light_grid": "#EEF1F4",
    "paper": "#FFFFFF",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
}

FONT_FAMILY = "Arial, Helvetica, sans-serif"


class SvgCanvas:
    """Small SVG writer sufficient for reproducible paper diagrams."""

    def __init__(
        self,
        width: float,
        height: float,
        *,
        title: str,
        description: str,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.description = description
        self.parts: list[str] = []

    def raw(self, fragment: str) -> None:
        self.parts.append(fragment)

    def group_start(self, transform: str | None = None) -> None:
        suffix = f' transform="{escape(transform)}"' if transform else ""
        self.parts.append(f"<g{suffix}>")

    def group_end(self) -> None:
        self.parts.append("</g>")

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        rx: float = 0.0,
        opacity: float = 1.0,
        dash: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
            f'height="{height:.2f}" rx="{rx:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
            f'opacity="{opacity:.3f}"{dash_attr}/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = PALETTE["ink"],
        stroke_width: float = 1.5,
        dash: str | None = None,
        opacity: float = 1.0,
        marker_end: str | None = None,
        linecap: str = "round",
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#{marker_end})"' if marker_end else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
            f'opacity="{opacity:.3f}" stroke-linecap="{linecap}"'
            f'{dash_attr}{marker_attr}/>'
        )

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 2.0,
        fill: str = "none",
        dash: str | None = None,
        opacity: float = 1.0,
        marker_end: str | None = None,
    ) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#{marker_end})"' if marker_end else ""
        self.parts.append(
            f'<polyline points="{encoded}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" '
            f'stroke-linejoin="round" stroke-linecap="round"'
            f'{dash_attr}{marker_attr}/>'
        )

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str = PALETTE["ink"],
        stroke_width: float = 1.5,
        dash: str | None = None,
        opacity: float = 1.0,
        marker_end: str | None = None,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#{marker_end})"' if marker_end else ""
        self.parts.append(
            f'<path d="{escape(d)}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" '
            f'stroke-linecap="round" stroke-linejoin="round"'
            f'{dash_attr}{marker_attr}/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str = "none",
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        self.parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
            f'opacity="{opacity:.3f}"/>'
        )

    def polygon(
        self,
        points: Iterable[tuple[float, float]],
        *,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: float = 14.0,
        fill: str = PALETTE["ink"],
        weight: int = 400,
        anchor: str = "start",
        style: str = "normal",
        letter_spacing: float | None = None,
        rotate: float | None = None,
    ) -> None:
        spacing_attr = (
            f' letter-spacing="{letter_spacing:.2f}"'
            if letter_spacing is not None
            else ""
        )
        transform_attr = (
            f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"'
            if rotate is not None
            else ""
        )
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" '
            f'font-family="{FONT_FAMILY}" font-size="{size:.2f}" '
            f'font-weight="{weight}" font-style="{style}" '
            f'text-anchor="{anchor}"{spacing_attr}{transform_attr}>'
            f'{escape(content)}</text>'
        )

    def marker(self, x: float, y: float, *, shape: str, colour: str) -> None:
        if shape == "circle":
            self.circle(x, y, 6.5, fill=colour, stroke=PALETTE["paper"], stroke_width=1.5)
        elif shape == "square":
            self.rect(
                x - 6.2,
                y - 6.2,
                12.4,
                12.4,
                fill=colour,
                stroke=PALETTE["paper"],
                stroke_width=1.5,
                rx=0.8,
            )
        elif shape == "diamond":
            self.polygon(
                ((x, y - 7.5), (x + 7.5, y), (x, y + 7.5), (x - 7.5, y)),
                fill=colour,
                stroke=PALETTE["paper"],
                stroke_width=1.5,
            )
        else:
            raise ValueError(f"unsupported marker shape {shape!r}")

    def render(self) -> str:
        definitions = f"""
<defs>
  <marker id="arrow-ink" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE['ink']}"/>
  </marker>
  <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE['blue']}"/>
  </marker>
  <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE['green']}"/>
  </marker>
  <pattern id="mask-hatch" width="9" height="9" patternUnits="userSpaceOnUse"
           patternTransform="rotate(45)">
    <rect width="9" height="9" fill="{PALETTE['light_grid']}"/>
    <line x1="0" y1="0" x2="0" y2="9" stroke="{PALETTE['grid']}"
          stroke-width="3"/>
  </pattern>
</defs>"""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" '
            f'height="{self.height:.0f}" viewBox="0 0 {self.width:.0f} '
            f'{self.height:.0f}" role="img" aria-labelledby="svg-title svg-desc">\n'
            f'<title id="svg-title">{escape(self.title)}</title>\n'
            f'<desc id="svg-desc">{escape(self.description)}</desc>\n'
            f'{definitions}\n'
            f'<rect x="0" y="0" width="{self.width:.0f}" height="{self.height:.0f}" '
            f'fill="{PALETTE["paper"]}"/>\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )
