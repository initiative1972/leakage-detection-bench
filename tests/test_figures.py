from leakbench import figures


def _valid(svg):
    return svg.startswith("<svg") and svg.rstrip().endswith("</svg>")


def test_leaderboard_svg():
    rows = [
        {"solver": "heuristic", "aggregate": 0.93},
        {"solver": "keyword_naive", "aggregate": 0.88},
    ]
    svg = figures.leaderboard_svg(rows)
    assert _valid(svg)
    assert "heuristic" in svg


def test_recall_by_type_svg():
    series = {
        "heuristic": {"target_leakage": 1.0, "group_leakage": 0.75},
        "keyword_naive": {"target_leakage": 1.0, "group_leakage": 1.0},
    }
    svg = figures.recall_by_type_svg(series)
    assert _valid(svg)
    assert "target" in svg
