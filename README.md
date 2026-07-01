# Aquifer-test transformation uncertainty benchmark

This repository contains the Computers & Geosciences submission package and reproducibility scaffold for the manuscript:

**A reproducible benchmark for transformation uncertainty in aquifer-test parameter inference**

The code supports a computational hydrogeology workflow that quantifies how pumping-test drawdown records are transformed into apparent aquifer parameters under four analytical interpretation pathways. The public repository contains the manuscript source, analysis scripts, selected summary tables, figures, and a lightweight quick test. Larger generated benchmark and field-analysis tables are archived in HydroShare.

## Repository layout

- `main.tex`: CAGEO CAS single-column manuscript source.
- `main.pdf`: compiled manuscript PDF, without cover letter or highlights.
- `cover_letter_CAGEO.tex` / `cover_letter_CAGEO.pdf`: cover letter source and PDF.
- `highlights_CAGEO.tex` / `highlights_CAGEO.pdf`: highlights source and PDF.
- `authorship_statement_CAGEO.txt`: CRediT authorship statement for upload.
- `supplementary_material.tex` / `supplementary_material.pdf`: supplementary material source and PDF.
- `analysis/`: Python scripts for analytical fitting, numerical benchmark postprocessing, figure generation, and QA.
- `examples/quick_test.py`: lightweight repository compliance and reproducibility test.
- `tables/`: selected small summary tables used by the quick test and manuscript figures.
- `fig*.pdf`: manuscript and supplementary figures.
- `LICENSE`: MIT license for the code.

## Quick test

The quick test does not rerun the full 10,000-scenario MODFLOW 6 benchmark. It verifies that the shipped summary tables contain the expected benchmark, transfer-regression, and field-management outputs.

Run from this folder:

```powershell
python examples/quick_test.py
```

Expected output:

```text
CAGEO quick test passed.
Quality-control rows: 40
Regression targets: p90_abs_lnM_T=0.48, p90_abs_lnM_S=0.51, p90_abs_lnM_response_time=0.55
Field cases: Lovelock Valley, Massachusetts
Screened model-factor rows: 40
```

## Installation for full workflows

Use Python 3. The analysis scripts use NumPy, pandas, SciPy, scikit-learn, matplotlib, cmcrameri, FloPy, and Pillow. MODFLOW 6 is required only for rerunning the numerical groundwater-flow simulations.

The lightweight quick test uses only the Python standard library.

## Build the manuscript

Run from this folder:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode cover_letter_CAGEO.tex
pdflatex -interaction=nonstopmode highlights_CAGEO.tex
pdflatex -interaction=nonstopmode supplementary_material.tex
pdflatex -interaction=nonstopmode supplementary_material.tex
```

The manuscript uses the CAGEO CAS single-column template with author-date references through `cas-model2-names.bst`.

## Full archive

The full reproducible data and software archive is available in HydroShare:

https://www.hydroshare.org/resource/3e8de7ff140c4ffab03163e215bb6634/

The HydroShare archive contains the larger MODFLOW 6 benchmark tables, field diagnostic outputs, and generated results used by the manuscript.

## License

Code in this repository is released under the MIT License. Data products archived in HydroShare are shared under CC BY 4.0 unless otherwise noted by their original source.
