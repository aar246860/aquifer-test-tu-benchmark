# Aquifer-test transformation uncertainty benchmark

This repository contains the Computers & Geosciences submission package and reproducibility scaffold for the manuscript:

**A reproducible benchmark for transformation uncertainty in aquifer-test parameter inference**

## Contents

- `main.tex`: Computers & Geosciences CAS-template manuscript source.
- `main.pdf`: compiled manuscript PDF, after local build.
- `cover_letter_CAGEO.tex` / `cover_letter_CAGEO.pdf`: cover letter source and compiled PDF.
- `highlights_CAGEO.txt`: journal highlights.
- `authorship_statement_CAGEO.txt`: CRediT authorship statement.
- `data_and_code_availability_CAGEO.txt`: submission data/code statement.
- `supplementary_material.tex` / `supplementary_material.pdf`: supplementary material source and PDF.
- `analysis/`: Python scripts used for analytical fitting, numerical benchmark postprocessing, figure generation, and QA.
- `tables/`: selected small summary tables needed for quick inspection. Full generated tables are hosted in HydroShare because several benchmark outputs exceed normal GitHub file-size limits.
- `fig*.pdf`: manuscript and supplementary figures.

## Full archive

The full reproducible data and software archive is available in HydroShare:

https://www.hydroshare.org/resource/3e8de7ff140c4ffab03163e215bb6634/

The HydroShare archive contains the larger MODFLOW 6 benchmark tables, field diagnostic outputs, and generated results used by the manuscript.

## Build the manuscript

Run from this folder:

```powershell
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplementary_material.tex
pdflatex -interaction=nonstopmode cover_letter_CAGEO.tex
```

The C&G manuscript uses the official CAS single-column template files: `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst`.

## Core dependencies

The Python workflow uses Python 3 with NumPy, pandas, SciPy, scikit-learn, matplotlib, cmcrameri, FloPy, and Pillow. MODFLOW 6 is required for rerunning the numerical groundwater-flow simulations.

## License

Code in this repository is released under the MIT License. Data products archived in HydroShare are shared under CC BY 4.0 unless otherwise noted by their original source.
