#!/bin/bash
#SBATCH --account=azh5924_b
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=hadjimichael@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=12GB

module load anaconda3/2021.05
source activate /storage/home/azh5924/.conda/envs/ABM_SA

python time_varying_means_adaptation_fraction.py './EU_het_2_outputs' $1