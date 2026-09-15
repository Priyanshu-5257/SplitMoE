#!/usr/bin/env python3
"""Build a self-contained, upload-ready arXiv source bundle."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ARXIV = ROOT / "arxiv"
SUBMISSION = ARXIV / "submission"
FIGURES = SUBMISSION / "figures"
ARCHIVE = ARXIV / "splitmoe-arxiv.tar.gz"

FIGURE_SOURCES = {
    "architecture.png": ROOT / "results" / "architecture.png",
    "quality_vs_parameters.png": ROOT / "results" / "frontier" / "quality_vs_parameters.png",
    "output_scale_factorial.png": ROOT / "results" / "reviewer_controls" / "output_scale_factorial.png",
    "top4_parameter_frontier.png": ROOT / "results" / "paper_scaling" / "top4_parameter_frontier.png",
    "top1_lambada.png": ROOT / "results" / "heldout" / "top1_lambada.png",
    "width_controlled_similarity.png": ROOT / "results" / "mechanism" / "width_controlled_similarity.png",
    "standardized_t4.png": ROOT / "results" / "systems" / "standardized_t4.png",
}


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required arXiv input is missing: {path}")


def main() -> None:
    sources = [PAPER / "main.tex", PAPER / "main.bbl", PAPER / "references.bib"]
    sources.extend(FIGURE_SOURCES.values())
    for source in sources:
        require(source)

    FIGURES.mkdir(parents=True, exist_ok=True)

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    graphicspath_start = r"\graphicspath{{../results/}"
    if graphicspath_start not in tex:
        raise RuntimeError("Canonical paper graphicspath did not match the expected layout")
    graphicspath_line = next(line for line in tex.splitlines() if line.startswith(r"\graphicspath"))
    tex = tex.replace(graphicspath_line, r"\graphicspath{{figures/}}", 1)
    (SUBMISSION / "main.tex").write_text(tex, encoding="utf-8")

    shutil.copy2(PAPER / "main.bbl", SUBMISSION / "main.bbl")
    shutil.copy2(PAPER / "references.bib", SUBMISSION / "references.bib")
    for output_name, source in FIGURE_SOURCES.items():
        shutil.copy2(source, FIGURES / output_name)

    expected = {
        "main.tex",
        "main.bbl",
        "references.bib",
        *(f"figures/{name}" for name in FIGURE_SOURCES),
    }
    actual = {
        path.relative_to(SUBMISSION).as_posix()
        for path in SUBMISSION.rglob("*")
        if path.is_file()
    }
    stale = actual - expected
    if stale:
        raise RuntimeError(f"Unexpected files in generated submission directory: {sorted(stale)}")
    missing = expected - actual
    if missing:
        raise RuntimeError(f"Generated submission is incomplete: {sorted(missing)}")

    ARXIV.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz") as bundle:
        for relative_path in sorted(expected):
            bundle.add(SUBMISSION / relative_path, arcname=relative_path)

    print(f"Built {SUBMISSION.relative_to(ROOT)}/ ({len(expected)} files)")
    print(f"Built {ARCHIVE.relative_to(ROOT)} ({ARCHIVE.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
