# arXiv submission package

Use `mauryac198@gmail.com` for the arXiv account and manuscript correspondence. The IITR address should only be used if it remains active and the affiliation is current; this manuscript identifies the author as an independent researcher.

## Build and verify

From the repository root:

```bash
python scripts/build_arxiv_bundle.py
```

Upload `splitmoe-arxiv.tar.gz` to arXiv. It contains a flat submission root with `main.tex`, the bibliography sources, and the seven required figures. Do not upload `paper/main.pdf` as part of a TeX source submission; inspect arXiv's compiled preview instead.

Before submitting:

1. Copy the fields from `METADATA.md` into arXiv's submission form.
2. Confirm the author name and current affiliation are accurate.
3. Select a license deliberately; it cannot be changed later.
4. Add the public repository URL to the Comments field.
5. Inspect every page of arXiv's generated PDF before selecting Submit.
6. If arXiv requests category endorsement, complete that process before the submission deadline.

The package is generated from `paper/main.tex`; edit the canonical paper and rebuild rather than editing `submission/main.tex` directly.
