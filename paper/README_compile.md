# How to Compile the Paper

## Quick Option (Recommended) — Overleaf
1. Go to https://www.overleaf.com and create a free account
2. Click "New Project" → "Upload Project"
3. Zip the entire `paper/` folder and upload it
4. Overleaf will auto-detect `main.tex` and compile to PDF
5. Set compiler to **pdfLaTeX** in the menu (top-left)

## Local Option — Install MacTeX
```bash
brew install --cask mactex
# After install, reload terminal:
eval "$(/usr/libexec/path_helper)"
# Then compile:
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # run twice for cross-references
```

## Files in This Package
| File | Description |
|------|-------------|
| `main.tex` | Full IEEE-format paper |
| `references.bib` | All 20 BibTeX references |
| `analysis.py` | Python script that generates all figures |
| `figures/fig1_*.pdf` | Dataset overview (4-panel) |
| `figures/fig2_*.pdf` | Confusion matrix |
| `figures/fig3_*.pdf` | Multi-class ROC curves |
| `figures/fig4_*.pdf` | Feature importance (XGB + LGB + Ensemble) |
| `figures/fig5_*.pdf` | Comparative model evaluation |
| `figures/fig6_*.pdf` | Learning curves + per-class F1 |
| `figures/fig7_*.pdf` | Calibration curves + IoT alert stats |
| `figures/fig8_*.pdf` | System architecture diagram |
| `figures/fig9_*.pdf` | 10-fold cross-validation |
| `figures/fig10_*.pdf` | Vital distributions per disease |
| `figures/fig11_*.pdf` | SHAP feature attribution |
| `figures/summary_stats.json` | All numeric results in JSON |
| `figures/model_metrics.csv` | Model comparison table |
| `figures/classification_report.csv` | Per-class metrics |

## Target Journals
- **IEEE Journal of Biomedical and Health Informatics (JBHI)** — IF 7.7
- **Computers in Biology and Medicine** — IF 7.0
- **Future Generation Computer Systems** — IF 6.2
- **Journal of Biomedical Informatics** — IF 4.5
