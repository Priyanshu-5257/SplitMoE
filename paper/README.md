# Paper build

The manuscript is a self-contained, venue-neutral preprint draft. From this directory, build it with:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Figures and raw tables are generated from committed JSON/CSV evidence under `results/`. Re-run the exporters from the repository root before rebuilding the paper.

The release contains:

- `main.tex` and the compiled `main.pdf`;
- `references.bib` with primary-source citations;
- `RELATED_WORK.md`, which records the novelty boundary against shared-expert and shared-base work;
- paired raw metrics and generated figures under `results/five_seed`, `results/frontier`, `results/mechanism`, `results/paper_scaling`, `results/heldout`, and `results/systems`.

This is a venue-neutral preprint draft. Submission-specific formatting, author metadata, and archival checkpoint hosting remain separate release decisions.

## arXiv bundle

From the repository root, create the self-contained arXiv source directory and upload archive with:

```bash
python scripts/build_arxiv_bundle.py
```

This writes `arxiv/submission/` and `arxiv/splitmoe-arxiv.tar.gz`. The generated source uses only local figure paths and includes both `references.bib` and the compiled `main.bbl`.
