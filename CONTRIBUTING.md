# Contributing

Thank you for helping improve the book. Typo fixes, corrected numbers, clearer derivations, better figures and new exercises are all welcome. For anything larger than a local fix (a new section, a restructured proof, a new case study), please open an issue first so we can agree on where it belongs.

## Writing rules

- Plain scholarly English in the first person plural ("we").
- Sentences of at most 32 words.
- Use commas, colons, parentheses or new sentences instead of em or en dashes. En dashes are allowed only in number ranges (`1980--2018`).
- Do not chain clauses or proof steps with semicolons.
- Define every technical term at first use with `\emph{}`.
- Write every fraction as `\frac{a}{b}`, including in figure labels. Never write `a/b` in math.
- The book is self-contained: do not refer to lectures, slides or problem sets.
- Every number taken from a paper must be checked against the paper, with the table or page cited, e.g. `\citet[Table~3]{Nevo2001}`.

## LaTeX conventions

- Do not add packages in chapter files. If you need one, add it to `preamble.tex` and say why in the pull request.
- Labels carry the chapter number: `sec:NN-...`, `eq:NN-...`, `fig:NN-...`, `tab:NN-...`, `prop:NN-...`. Cross-reference with `\cref`.
- Put figure files (PDF or PNG) in `figures/chNN/`.
- Add or correct references in `references.bib`. Citation keys are the authors' surnames (up to three) plus the year, e.g. `BerryLevinsohnPakes1995`; use `FirstEtAlYear` for four or more authors. Reuse an existing key if the work is already there.
- Keep the sources free of LaTeX comments.

## Before opening a pull request

1. Build the book (see the README) and confirm that the log has no errors, no undefined references or citations, and no overfull boxes wider than 10pt.
2. Check new or changed text against the writing rules above.
3. Describe what you changed and, for factual corrections, the source (paper, table or page).

You may commit the rebuilt `IO_Textbook.pdf` with your change or leave it to the maintainer.

## License of contributions

The book is licensed under CC BY-SA 4.0 (see `LICENSE`). By opening a pull request you agree that your contribution is licensed under the same terms.
