# Empirical Industrial Organization: A Course Companion for ECON 257

A self-contained graduate textbook in empirical industrial organization: market power, demand estimation, mergers, collusion and auctions. Each chapter develops the theory with full derivations and then works through the main empirical papers in depth: setting, data, model, estimating equations, identification, results and critiques.

**Read the book:** [`IO_Textbook.pdf`](IO_Textbook.pdf).

## Contents

| Part | Chapter | Source file |
|---|---|---|
| I. Market Power | 1. Introduction: What Empirical IO Studies | `chapters/ch01_introduction.tex` |
| | 2. The Theory of Pricing and Market Power | `chapters/ch02_pricing_theory.tex` |
| | 3. Measuring Market Power in Homogeneous-Goods Industries | `chapters/ch03_measuring_market_power.tex` |
| | 4. Oligopoly with Differentiated Products | `chapters/ch04_differentiated_oligopoly.tex` |
| II. Demand Estimation | 5. Demand in Product Space: The Almost Ideal Demand System | `chapters/ch05_aids.tex` |
| | 6. Discrete Choice Foundations and the Vertical Model | `chapters/ch06_discrete_choice.tex` |
| | 7. Logit, Nested Logit, and the Berry Inversion | `chapters/ch07_logit.tex` |
| | 8. Random Coefficients Logit: The BLP Model | `chapters/ch08_blp.tex` |
| | 9. Airlines: Hubs, Entry, and Market Power | `chapters/ch09_airlines.tex` |
| | 10. Valuing New Goods | `chapters/ch10_new_goods.tex` |
| III. Competition Policy | 11. Horizontal Mergers | `chapters/ch11_mergers.tex` |
| | 12. Collusion: Theory | `chapters/ch12_collusion_theory.tex` |
| | 13. Collusion: Evidence and Detection | `chapters/ch13_collusion_evidence.tex` |
| IV. Auctions | 14. Auction Theory for Single-Object Auctions | `chapters/ch14_auction_theory.tex` |
| | 15. Identification and Estimation of Single-Object Auctions | `chapters/ch15_auction_empirics.tex` |
| Appendices | A. Econometric Toolkit | `chapters/appA_econometrics.tex` |
| | B. Mathematical Toolkit | `chapters/appB_math.tex` |
| | C. Worked Problems | `chapters/appC_problems.tex` |

## Repository layout

```
IO_Textbook.tex    master file (title page, parts, chapters, bibliography)
IO_Textbook.pdf    compiled book
preamble.tex       the only preamble: packages, theorem environments, macros
chapters/          chapter files chNN_<slug>.tex; each inputs its parts from chapters/chNN/partK.tex
figures/chNN/      figures and table files included by chapter NN
figures/chNN/src/  Python code (and data) that produces them and the worked numbers in the text
references.bib     bibliography
requirements.txt   Python packages the scripts in figures/*/src/ need
```

## Building the PDF

You need a TeX distribution (TeX Live or MiKTeX) with `pdflatex` and `bibtex`. From the repository root:

```
pdflatex IO_Textbook.tex
bibtex IO_Textbook
pdflatex IO_Textbook.tex
pdflatex IO_Textbook.tex
```

or simply `latexmk -pdf IO_Textbook.tex`.

The figures are committed, so you do not need Python to build the book.

## Regenerating figures and numbers

To regenerate a figure or a number, run its script in `figures/chNN/src/` (or `figures/appX/src/` for the appendices). The scripts need Python 3.11 or later (tested with Python 3.13) and the six packages in [`requirements.txt`](requirements.txt): numpy, scipy, pandas, matplotlib, sympy and statsmodels. Install them with

```
pip install -r requirements.txt
```

The 25-product logit market used by `figures/ch04/src/logit25.py`, `logit_uniqueness_check.py`, `supply_examples.py` and `figures/ch07/src/logit_pricing_example.py` is the committed file `figures/appC/data/logit_market.csv`, which the scripts read by a path relative to their own location. In the same way, `figures/appA/src/sumner_inference.py` and `figures/appC/src/sumner_passthrough.py` read the committed cigarette panel `figures/appC/data/cigarette_panel.csv`.

Most scripts use only numpy, scipy, pandas and matplotlib. The others need the following.

- sympy: `figures/appA/src/kernel_constants.py`, `figures/ch12/src/chenrey_check.py` and, in `figures/ch14/src/`, `apv_example.py`, `asym_uniform.py`, `collusion_example.py` and `part2_numbers.py`.
- statsmodels: `figures/ch03/src/sumner_replication.py`. In `figures/appC/src/sumner_passthrough.py` it runs only an optional cross-check, which the script skips when statsmodels is missing.
- linearmodels (optional, not in `requirements.txt`): `figures/appC/src/appC_vertical_logit.py` uses it to cross-check its hand-coded clustered 2SLS. Without it, the script records `None` for that check and leaves every other result unchanged.
- A TeX installation: `figures/ch13/src/asker_inverse_bid.py` and `figures/ch13/src/bm_static.py` typeset their labels with LaTeX (matplotlib's `text.usetex`). They need `latex` and `dvipng` on the path and the newpx packages (newpxtext and newpxmath) in the TeX installation.

## Contributing

Corrections and improvements are welcome through pull requests. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Copyright © 2026 Angikar Ghosal and contributors.

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0); the full text is in [`LICENSE`](LICENSE). You may share and adapt it as long as you give credit and license your changes under the same terms. By submitting a pull request, you agree that your contribution is licensed under CC BY-SA 4.0.
