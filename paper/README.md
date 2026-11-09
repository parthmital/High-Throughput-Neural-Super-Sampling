# Paper

Two documents on the whole NeuralSS project for the **Journal of Real-Time Image Processing** (Springer), from one set of sources:

- **Article** (`main.tex`): an original research article within the journal's 12-page limit. It states the design and the main results and points to the supplement for detail.
- **Supplementary material** (`supplementary.tex`, Online Resource 1): no page limit. It explains every concept, equation, input, loss, protocol and result from first principles and shows every table, plot and prediction of the three notebooks with its interpretation.

The notebooks are an intermediate checkpoint, so both documents are built to be revised after every run. No run number is typed by hand, figures in the supplement are read straight from the notebooks' output folders, and every finding carries a status label. The runs reported now are listed in Section S1.3 of the supplement.

## Files

| File                                          | Purpose                                                                                                                          |
| :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| `main.tex`                                    | Article source (Springer Nature LaTeX template v3.1, December 2024, two-column `[iicol]` as the journal recommends)              |
| `main-anonymised.tex`                         | Anonymised article: no author name, e-mail, affiliation or repository link                                                       |
| `main-title-page.tex`                         | Separate title page: title, author, affiliation, abstract, keywords and the corresponding author                                 |
| `main-no-references.tex`                      | Similarity-check copy of the article without the reference list                                                                  |
| `supplementary.tex`                           | Supplement wrapper: title block with the journal, article and corresponding author (as the journal asks), S-numbering            |
| `supplementary-anonymised.tex`                | Anonymised supplement, paired with `main-anonymised.tex`                                                                         |
| `supp-overview.tex`                           | S1: purpose, reading guide, runs reported, status labels, conventions                                                            |
| `supp-background.tex`                         | S2: fundamentals (sampling, motion, warping, TAA, network building blocks, interpolation, latency, every metric)                 |
| `supp-data.tex`                               | S3: data sources, filtering, copy detection, calibration, leakage-safe split, audit, synthetic renderer                          |
| `supp-tsr-method.tex`, `supp-tsr-results.tex` | S4, S5: temporal super-resolution method and results                                                                             |
| `supp-fg-method.tex`, `supp-fg-results.tex`   | S6, S7: frame-generation method, and its results with real-time cost and pacing                                                  |
| `supp-status.tex`                             | S8: register of findings with status, defects found in the code of the reported runs, limitations, planned revisions             |
| `supp-appendix.tex`                           | S9: run provenance, hardware, stage times, full configurations, reproducibility, revision procedure                              |
| `paper-preamble.tex`                          | Packages and macros shared by both documents: `\Res` value lookup, status labels, variant suffix                                 |
| `results-values.tex`                          | Generated value of every result key (`tools/build_paper_results.py`). Committed, because the journal upload needs it. Never edit |
| `references.bib`                              | References of the article, each checked against its DOI or publisher page and the passage it supports                            |
| `references-supplementary.bib`                | References cited only in the supplement, checked the same way                                                                    |
| `sn-jnl.cls`, `sn-mathphys-num.bst`           | Unmodified template class and numbered reference style                                                                           |
| `fig-*.png`                                   | Article figures: byte copies or exact crops of the notebooks' saved outputs, recreated by the generator                          |
| `cover-letter.md`                             | Cover letter text (the journal asks it to name the real-time issues addressed)                                                   |
| `*.pdf`                                       | Built outputs of the six sources                                                                                                 |

The journal asks for a flat upload with no subfolders, so the article's figures sit next to `main.tex`. The supplement is uploaded as one PDF (`supplementary.pdf`, renamed `ESM_1.pdf` on upload) and is referred to as "Online Resource 1".

## Writing conventions

- **Numbers.** Write `\Res{key}` for any number from a run, `\ResDiff{a}{b}`, `\ResAbsDiff{a}{b}` and `\ResRatio{a}{b}` for derived ones. Keys are listed in `results-values.tex` (grep it). A missing key prints a red `??key` and the generator exits 1.
- **Status.** Tag every result claim in the supplement with `\Confirmed`, `\Preliminary`, `\Observation` or `\UnderRevision` (defined in Section S1.4). Section S8 is the register of all findings.
- **Cross-references.** The article cites supplement labels with the prefix `S-` (`\cref{S-sec:tsrm}`) and the supplement cites article labels with `M-`, through `xr-hyper`.
- **Citations.** Only peer-reviewed, verified references. A place that needs one gets `\NeedsCite{...}`, which prints red so it cannot survive a final build.
- **Single-section test build.** Create a git-ignored `test-<name>.tex` containing `\def\SuppOnly{supp-<name>}\input{supplementary}` and build it with Tectonic, so parallel builds do not collide.

## Build

One command, from the repository root, regenerates `results-values.tex` and the article figures from the notebooks' outputs, then builds the six PDFs with Tectonic (`paper/tectonic.exe`, else `tectonic` on PATH) in the order `xr-hyper` needs (article, supplement, article, for each pair):

```sh
.venv/Scripts/python -I tools/build_paper.py                  # all six PDFs
.venv/Scripts/python -I tools/build_paper.py --supplement-only
.venv/Scripts/python -I tools/build_paper.py --no-generate    # keep the current values
```

`tools/compare_fg_tiers_middlebury.py` writes `neural-frame-generation/local-evaluation/middlebury_psnr.csv`, which the generator reads. Run it on a CPU after new FG weights.

## Revising after a new run

1. Download the run's outputs into the notebook's `outputs/` folder (and rerun the Middlebury comparison after an FG run).
2. Run `tools/build_paper_results.py`. It prints every key that was added, removed or changed, with the `file:line` of each sentence that uses a changed or removed key.
3. Re-read each listed sentence: is the comparison still true, is the status label still right? Update the run list in `supp-overview.tex`, the register in `supp-status.tex` (move fixed defects out, add new findings), and any description of code that changed.
4. Rebuild with `tools/build_paper.py`, check that `main.pdf` and `main-anonymised.pdf` are still at most 12 pages, and run the `latex-research-papers` checker on `main.tex` and `supplementary.tex`.

The full procedure is in Section S9 of the supplement.

## Submission notes (journal guidelines, read 8 November 2026)

- 12 double-column pages at most, including references and any author biography (the article is 12 pages). Abstract 150 to 250 words, 4 to 6 keywords (6), numbered references in square brackets, "Statements and Declarations" before the references, single-blind review.
- Upload the LaTeX source with the class, style, `paper-preamble.tex`, `results-values.tex` and figures, and the compiled PDF.
- Supplementary material: name the file consecutively (`ESM_1.pdf`), refer to it as "Online Resource 1", include the article title, journal name, author names and the corresponding author's affiliation and e-mail in the file (the supplement's title block does), and give a short caption in the submission system. It is published as received, without editing.
- Author contributions and competing interests are also entered in the submission system.
- To confirm before submitting: the funding and competing-interest statements, an ORCID if available, and whether the journal wants a short author biography (it would count towards the 12 pages).
