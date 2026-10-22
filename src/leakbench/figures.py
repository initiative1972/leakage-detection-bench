"""Dependency-free SVG figures: a leaderboard bar chart and a per-type recall
chart. No matplotlib, so CI stays lean.
"""
from __future__ import annotations

from typing import Dict, List

W, H = 560, 360
ML, MR, MT, MB = 150, 20, 40, 40
PW = W - ML - MR
PH = H - MT - MB

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _header(title: str) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{W/2}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="bold">{title}</text>',
    ]


def leaderboard_svg(rows: List[dict]) -> str:
    """Horizontal bars of aggregate score per solver (sorted as given)."""
    out = _header("LeakBench leaderboard (aggregate score)")
    n = len(rows)
    bar_h = PH / (n + 0.5)
    for i, r in enumerate(rows):
        agg = r["aggregate"]
        cy = MT + i * bar_h
        out.append(f'<text x="{ML-8}" y="{cy+bar_h*0.55:.1f}" text-anchor="end" '
                   f'font-size="11">{r["solver"]}</text>')
        out.append(f'<rect x="{ML}" y="{cy+bar_h*0.2:.1f}" '
                   f'width="{agg*PW:.1f}" height="{bar_h*0.6:.1f}" '
                   f'fill="{_PALETTE[i % len(_PALETTE)]}"/>')
        out.append(f'<text x="{ML+agg*PW+4:.1f}" y="{cy+bar_h*0.6:.1f}" '
                   f'font-size="10">{agg:.3f}</text>')
    out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" '
               f'stroke="black"/>')
    out.append(f'<text x="{ML+PW/2}" y="{H-10}" text-anchor="middle" '
               f'font-size="11">aggregate = 0.5*F1 + 0.3*type + 0.2*localisation'
               f'</text>')
    out.append("</svg>")
    return "\n".join(out)


def recall_by_type_svg(series: Dict[str, Dict[str, float]]) -> str:
    """Grouped bars: per-leak-type detection recall for each named solver."""
    out = _header("Detection recall by leak type")
    types = sorted({t for s in series.values() for t in s})
    solvers = list(series.keys())
    n_types = len(types)
    group_w = PW / max(n_types, 1)
    bw = group_w / (len(solvers) + 1)

    for ti, t in enumerate(types):
        gx = ML + ti * group_w
        for si, s in enumerate(solvers):
            val = series[s].get(t, 0.0)
            x = gx + si * bw + bw * 0.5
            y = MT + (1 - val) * PH
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.9:.1f}" '
                       f'height="{val*PH:.1f}" '
                       f'fill="{_PALETTE[si % len(_PALETTE)]}"/>')
        label = t.replace("_leakage", "")
        out.append(f'<text x="{gx+group_w/2:.1f}" y="{MT+PH+14}" '
                   f'text-anchor="middle" font-size="9">{label}</text>')

    for si, s in enumerate(solvers):
        out.append(f'<rect x="{ML+si*110}" y="{MT-14}" width="10" height="10" '
                   f'fill="{_PALETTE[si % len(_PALETTE)]}"/>')
        out.append(f'<text x="{ML+si*110+14}" y="{MT-5}" font-size="10">'
                   f'{s}</text>')
    out.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" '
               f'stroke="black"/>')
    out.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" '
               f'stroke="black"/>')
    out.append("</svg>")
    return "\n".join(out)
