# Paper

Manuscript on the whole NeuralSS project for the **Journal of Real-Time Image Processing** (Springer), an original research article. It reports the data pipeline run of 18 October 2026, the TSR run of 3 to 4 November 2026 and the FG run of 8 November 2026, plus the Middlebury comparison measured locally on 8 November 2026.

## Files

| File                                | Purpose                                                                                                                      |
| :---------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| `main.tex`                          | Manuscript source (Springer Nature LaTeX template v3.1, December 2024, two-column `[iicol]` as the journal recommends)       |
| `main-anonymised.tex`               | Anonymised copy: no author name, e-mail, affiliation or repository link                                                      |
| `main-title-page.tex`               | Separate title page: title, author, affiliation, abstract, keywords and the corresponding author                             |
| `main-no-references.tex`            | Similarity-check copy without the reference list                                                                             |
| `references.bib`                    | 42 references, each checked against its DOI or publisher page and the passage it supports                                    |
| `sn-jnl.cls`, `sn-mathphys-num.bst` | Unmodified template class and numbered reference style                                                                       |
| `fig-*.png`                         | Figures: copies or panel crops of the notebooks' saved plots and predictions (`<notebook>/outputs/plots`, `.../predictions`) |
| `cover-letter.md`                   | Cover letter text (the journal asks it to name the real-time issues addressed)                                               |
| `*.pdf`                             | Built outputs of the four sources                                                                                            |

The journal asks for a flat upload with no subfolders, so the figures sit next to `main.tex`.

## Build

Tectonic (not committed, `tectonic.exe` in this folder is Git-ignored):

```sh
cd paper
./tectonic.exe --keep-logs --reruns 3 main.tex
./tectonic.exe --keep-logs --reruns 3 main-anonymised.tex
./tectonic.exe --keep-logs --reruns 3 main-title-page.tex
./tectonic.exe --keep-logs --reruns 3 main-no-references.tex
```

`references.bib` is generated from the verified entries by `.agent-local/scripts/build_paper_bib.py` (agent workspace, not committed). Edit it directly if that workspace is absent.

## Submission notes (journal guidelines, read 8 November 2026)

- 12 double-column pages at most, including references (the manuscript is 12 pages). Abstract 150 to 250 words (244), 4 to 6 keywords (6), numbered references in square brackets, "Statements and Declarations" before the references, single-blind review.
- Upload the LaTeX source with the class, style and figures and the compiled PDF.
- Author contributions and competing interests are also entered in the submission system.
- To confirm before submitting: the funding and competing-interest statements, an ORCID if available, and whether the journal wants a short author biography inside the 12 pages.
