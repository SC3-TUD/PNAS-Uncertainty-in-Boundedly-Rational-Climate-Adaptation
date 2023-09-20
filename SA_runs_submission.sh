#!/bin/bash
#SBATCH --account=azh5924_b
#SBATCH --partition=sla-prio
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=4:00:00
#SBATCH --mail-user=hadjimichael@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH --exclusive=mcs
#SBATCH --mem=0

module load anaconda3/2021.05
source activate /storage/home/azh5924/.conda/envs/ABM_SA
module load parallel

srun="srun -n 1 -c $SLURM_CPUS_ON_NODE "
parallel="parallel --max-procs $SLURM_CPUS_ON_NODE --delay 0.2 --joblog sensitivity_analysis_$3.log --resume"
vals=($(seq $1 $2))
$srun $parallel "python3 single_run_for_SA.py" ::: "${vals[@]}" ::: {0..10}