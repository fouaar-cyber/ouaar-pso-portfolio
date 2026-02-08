# Utility-Driven Inertia in Particle Swarm Optimisation

Replication code for Ouaar & Ouaar (2025), "Utility-Driven Inertia in Particle Swarm Optimisation: A Reproducible Approach to Higher-Moment Portfolio Selection".


## Folder Structure

ouaar-pso-portfolio/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── revised_paper_code.py      # Main code
├── generate_figures.py         # Figure generation
├── main.tex                    # LaTeX paper
├── performance_table.tex       # Auto-generated table
├── data/
│   ├── summary_statistics.csv
│   └── sensitivity_results.npy
└── figures/
├── figure1_sharpe_comparison.pdf
├── figure2_cumulative_returns.pdf
├── figure3_omega_evolution.pdf
├── figure4_sharpe_distribution.pdf
├── figure5_transaction_cost.pdf
└── sensitivity_heatmap.pdf

## Requirements

- Python 3.11+
- NumPy, pandas, scikit-learn, yfinance, matplotlib, seaborn

## Install dependencies:

pip install -r requirements.txt

## Running the Code:

python revised_paper_code.py

## Generate figures:

python generate_figures.py

##  Data

S&P 500 and BIST-100 data are downloaded automatically via yfinance. No manual data download required.

##  Output

    summary_statistics.csv — Performance metrics
    performance_table.tex — LaTeX results table
    figure*.pdf — All paper figures
    sensitivity_heatmap.pdf — Utility weight sensitivity

##  Citation: If using this code, please cite:

Ouaar, S., & Ouaar, F. (2025). Utility-Driven Inertia in Particle Swarm Optimisation: 
A Reproducible Approach to Higher-Moment Portfolio Selection. 
[Journal name], [Volume], [Pages].

##  License
MIT License — feel free to use and modify with attribution.