# Utility-Driven Inertia in Particle Swarm Optimisation

[![DOI](https://zenodo.org/badge/YOUR_REPO_ID.svg)](https://doi.org/10.5281/zenodo.18846493)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

Replication code for: **Ouaar, S., & Ouaar, F.** "Utility-Driven Inertia in Particle Swarm Optimisation: A Reproducible Approach to Higher-Moment Portfolio Selection."

## Authors

- **Fatima Ouaar** (Corresponding author) — Department of Mathematics, University of Biskra, Algeria  
  Email: f.ouaar@univ-biskra.dz
- **Safia Ouaar** — Department of Economics, University of Biskra, Algeria

## Repository Structure
├── author_contributions.txt
├── CITATION.cff
├── conflict_of_interest.txt
├── data_availability.txt
├── figure1_sharpe_comparison.pdf
├── figure2_cumulative_returns.pdf
├── figure3_omega_evolution.pdf
├── figure4_sharpe_distribution.pdf
├── figure5_transaction_cost.pdf
├── generate_figures.py
├── .gitignore
├── highlights.txt
├── LICENSE
├── Ouaar paper.pdf
├── Ouaar paper.tex
├── performance_table.tex
├── references.bib
├── requirements.txt
├── revised_paper_code.py
├── sensitivity_heatmap.pdf
├── sensitivity_results.npy
├── summary_statistics.csv
└── README.md

### Manuscript
| File | Description |
|------|-------------|
| `Ouaar paper.tex` | LaTeX manuscript |
| `references.bib` | Bibliography |
| `Ouaar paper.pdf` | Compiled PDF |

### Code
| File | Description |
|------|-------------|
| `revised_paper_code.py` | Main PSO implementation |
| `generate_figures.py` | Figure generation script |
| `performance_table.tex` | LaTeX table output |
| `sensitivity_results.npy` | Sensitivity analysis results |
| `summary_statistics.csv` | Summary statistics |

### Figures
| File | Description |
|------|-------------|
| `figure1_sharpe_comparison.pdf` | Sharpe ratio comparison |
| `figure2_cumulative_returns.pdf` | Cumulative returns |
| `figure3_omega_evolution.pdf` | Omega evolution |
| `figure4_sharpe_distribution.pdf` | Sharpe distribution |
| `figure5_transaction_cost.pdf` | Transaction cost analysis |
| `sensitivity_heatmap.pdf` | Utility weight sensitivity |

### Submission Documents
| File | Description |
|------|-------------|
| `highlights.txt` | Paper highlights |
| `author_contributions.txt` | Author contributions |
| `conflict_of_interest.txt` | COI declaration |
| `data_availability.txt` | Data availability statement |

### Configuration
| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |
| `.gitignore` | Git ignore rules |
| `CITATION.cff` | Citation metadata |


## Installation

```bash
# Clone repository
git clone https://github.com/fouaar-cyber/ouaar-pso-portfolio.git
cd ouaar-pso-portfolio

# Install dependencies
pip install -r requirements.txt

## Usage
Run Main Analysis

## cd Code
python revised_paper_code.py

## Generate Figures

python generate_figures.py

## Compile LaTeX Manuscript


cd Manuscript
pdflatex ouaar_pso_portfolio.tex
bibtex ouaar_pso_portfolio
pdflatex ouaar_pso_portfolio.tex
pdflatex ouaar_pso_portfolio.tex

## Data
All data are downloaded automatically via yfinance:

    S&P 500: 51 liquid ETFs and stocks (2005–2023)
    BIST-100: 66 assets with USD-denominated returns (2010–2023)

No manual data download required.
Output Files
| File                      | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `summary_statistics.csv`  | Performance metrics across models                 |
| `performance_table.tex`   | LaTeX-formatted results table                     |
| `figure*.pdf`             | All paper figures (in root for LaTeX compilation) |
| `sensitivity_heatmap.pdf` | Utility weight sensitivity analysis               |


## Citation
If using this code, please cite:
@software{ouaar2026pso,
  title={Utility-Driven Inertia in Particle Swarm Optimisation: Replication Code},
  author={Ouaar, Safia and Ouaar, Fatima},
  year={2026},
  url={https://github.com/fouaar-cyber/ouaar-pso-portfolio},
  note={Replication code for "Utility-Driven Inertia in Particle Swarm Optimisation: A Reproducible Approach to Higher-Moment Portfolio Selection"}
}

## For the accompanying paper (once published):

@article{ouaar2026utility,
  title={Utility-Driven Inertia in Particle Swarm Optimisation: A Reproducible Approach to Higher-Moment Portfolio Selection},
  author={Ouaar, Safia and Ouaar, Fatima},
  year={2026},
  note={Manuscript submitted for publication}
}

## License
This project is licensed under the MIT License — see LICENSE file.
Contact

    Issues: GitHub Issues
    Email: f.ouaar@univ-biskra.dz

## Acknowledgments
We thank the open-source community for NumPy, pandas, scikit-learn, matplotlib, and yfinance.

