# Sanshodhak Journal Paper (LaTeX)

## Files
- `main.tex` — full journal-style manuscript draft
- `references.bib` — bibliography

## Build
Use either of the following:

```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or

```bash
cd /home/admin-/Desktop/Sanshodhak/paper-intel/paper
latexmk -pdf main.tex
```

## Notes
- The manuscript is written from implementation-grounded evidence in this repository.
- Metrics are reported with caveats where evaluation scripts differ.
- Before submission, replace `Anonymous Authors` and add affiliation details.
- If your target venue is not IEEE, migrate to the required class/template (ACM, Springer, Elsevier, etc.).
