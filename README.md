# Utility-Driven Inertia in Particle Swarm Optimisation

Replication code for Ouaar & Ouaar (2026), "Utility-Driven Inertia in Particle Swarm Optimisation: A Reproducible Approach to Higher-Moment Portfolio Selection".

## Authors

- Fatima Ouaar (Department of Mathematics, University of Biskra, Algeria) - Corresponding author: f.ouaar@univ-biskra.dz
- Safia Ouaar (Department of Economics, University of Biskra, Algeria)

## Repository Structure

├── Manuscript Files
│   ├── Ouaar paper.tex                    # LaTeX manuscript (Emerging Markets Review format)
│   ├── references.bib              # Bibliography (BibTeX)
│   ├── Ouaar paper.pdf                    # Compiled PDF
│   
│
├── Submission Documents
│   ├── highlights.txt              # Paper highlights (5 bullet points)
│   ├── author_contributions.txt    # Author contribution statement
│   ├── conflict_of_interest.txt    # Declaration of competing interests
│   └── data_availability.txt       # Data availability statement
│
├── Figures (PDF)
│   ├── figure1_sharpe_comparison.pdf
│   ├── figure2_cumulative_returns.pdf
│   ├── figure3_omega_evolution.pdf
│   ├── figure4_sharpe_distribution.pdf
│   ├── figure5_transaction_cost.pdf
│   └── sensitivity_heatmap.pdf
│
├── Python Code
│   ├── revised_paper_code.py       # Main PSO algorithm implementation
│   ├── generate_figures.py         # Figure generation script
│   ├── performance_table.tex       # LaTeX table output
│   ├── sensitivity_results.npy     # Sensitivity analysis results
│   └── summary_statistics.csv      # Summary statistics
│
└── Configuration
├── README.md                   # This file
├── LICENSE                     # License information
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules


## Compilation

To compile the LaTeX manuscript:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

## Data Availability
All data used in this study are publicly available:

    S&P 500: Yahoo Finance via yfinance Python library
    BIST-100: Yahoo Finance via yfinance Python library (ticker format .IS)

## Replication code: Python 3.11, NumPy, pandas, scikit-learn, matplotlib, yfinance
Installation
Install dependencies:
bash
pip install -r requirements.txt

## Running the Code
Run main analysis:
bash
python revised_paper_code.py

## Generate figures:
bash
python generate_figures.py

## Data
S&P 500 and BIST-100 data are downloaded automatically via yfinance. No manual data download required.

## Output

    summary_statistics.csv — Performance metrics
    performance_table.tex — LaTeX results table
    figure*.pdf — All paper figures
    sensitivity_heatmap.pdf — Utility weight sensitivity

## Citation
If using this code, please cite:

Ouaar, S., & Ouaar, F. (2025). Utility-Driven Inertia in Particle Swarm Optimisation: 
A Reproducible Approach to Higher-Moment Portfolio Selection. 


## Contact
For questions or issues, please contact:

    Fatima Ouaar: f.ouaar@univ-biskra.dz
    GitHub Issues: https://github.com/fouaar-cyber/ouaar-pso-portfolio/issues


