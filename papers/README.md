# Papers

This directory contains the source and released PDFs for our papers.
Each paper lives in its own directory:

```text
papers/
  README.md
  <paper-name>/
    main.tex
    references.bib
    figures/
    metadata.yaml
    CITATION.cff
```

`metadata.yaml` defines the paper title, authors, date, and status. `main.tex`
must read these values for its title block. `main.tex` is the authoritative
source. Commit `main.pdf` only when a paper is ready to share, so the repository
contains a directly readable release beside its source. Keep generated auxiliary
LaTeX files out of version control.
Unless stated otherwise in a paper directory, papers are licensed under
[CC-BY-4.0](LICENSE-CC-BY-4.0.md).

## Citation

Cite Binaml software through the repository-level
[CITATION.cff](../CITATION.cff). Cite a paper separately when its model or
benchmark specification informs your work, using that paper's `CITATION.cff`.

## Papers

- [`binaml`](binaml/): *Binaml: Continual Learning over Binary Feature
  Streams*;
  [citation metadata](binaml/CITATION.cff).
- [`synthetic-drifting-streams`](synthetic-drifting-streams/): *Synthetic
  Drifting Streams over Binary Features: Regression and Classification*;
  [citation metadata](synthetic-drifting-streams/CITATION.cff).
