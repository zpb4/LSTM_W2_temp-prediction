#!/bin/bash

#SBATCH -t 100:00:00
#SBATCH --job-name=lstm-grid
#SBATCH -p normal
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=lstm-grid.txt
#SBATCH --ntasks-per-node=80

module load python/3.11.5
source ~/py311-env/bin/activate

python ./src/lstm_outflow-temp-prediction_grid-search_90-24.py




