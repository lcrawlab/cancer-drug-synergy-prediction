#!/bin/bash

#SBATCH --mail-type=ALL                                     #email notification
#SBATCH --mail-user=alexandra_wong@brown.edu                #email address
#SBATCH -J 20250531_percgrow_mfpprot_svm_pgreg_experiment      #job name

#SBATCH --partition=bigmem                                  #partition
#SBATCH -N 1                                                #number of nodes
#SBATCH -c 2                                                #number of cores

#SBATCH -t 96:00:00                                         #time limit (HH:MM:SS)
#SBATCH --mem=150G                                          #memory per node

#SBATCH --array=1                                         #job array

# Use '%A' for job array ID, '%J' for job ID, '%a' for array task ID
#SBATCH --output=scratch/anticancer-synergy-prediction-scratch/experiments/20250531_percgrow_svm/mfpprot_svm_pgreg_%a_slurm.out           #output file
#SBATCH --error=scratch/anticancer-synergy-prediction-scratch/experiments/20250531_percgrow_svm/mfpprot_svm_pgreg_%a_slurm.err            #error file

source env/bin/activate

SCRATCH="scratch/anticancer-synergy-prediction-scratch/"
export PYTHONPATH="${PYTHONPATH}:~/cancer-drug-synergy-prediction/"
PYOF=$SCRATCH"experiments/20250531_percgrow_svm/mfpprot_svm_pgreg"$SLURM_ARRAY_TASK_ID".txt"
PLOT=$SCRATCH"experiments/20250531_percgrow_svm/mfpprot_svm_pgreg"

# Run hyperparameter experiments using job array index to select hyperparameters in the vnn_experiments_oscar.py script
# REMEMBER TO INDEX BY 1
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME starting on `date`"
python3 models/run/run_svm_models.py --score 'PERCENTGROWTH' --use_mfp --use_prot --use_pgreg --output_fp $PLOT"metrics.csv" > $PYOF
echo "Job $SLURM_ARRAY_TASK_ID ended at `date`"