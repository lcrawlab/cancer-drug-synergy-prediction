#!/bin/bash

#SBATCH --mail-type=ALL                                     #email notification
#SBATCH --mail-user=alexandra_wong@brown.edu                #email address
#SBATCH -J 20240825_modelXGBoost_targeted_other_mfpdna_bc0                    #job name

#SBATCH --partition=gpu                                     #gpu partition
#SBATCH --gres=gpu:1                                        #request 1 gpu
#SBATCH -N 1                                                #number of nodes
#SBATCH -c 2                                                #number of cores

#SBATCH -t 10:00:00                                         #time limit (HH:MM:SS)
#SBATCH --mem=15G                                           #memory per node

#SBATCH --array=1                                        #job array

# Use '%A' for job array ID, '%J' for job ID, '%a' for array task ID
#SBATCH --output=scratch/anticancer-synergy-prediction-scratch/experiments/20240825_modelXGBoost_targeted_other/bc0/mfpdna/slurm_%a.out           #output file
#SBATCH --error=scratch/anticancer-synergy-prediction-scratch/experiments/20240825_modelXGBoost_targeted_other/bc0/mfpdna/slurm_%a.err            #error file


source env/bin/activate

SCRATCH="scratch/anticancer-synergy-prediction-scratch/"

OF=$SCRATCH"experiments/20240825_modelXGBoost_targeted_other/bc0/mfpdna/"

# Run hyperparameter experiments using job array index to select hp_mfpdna_bc0 in the vnn_experiments_oscar.py script
# REMEMBER TO INDEX BY 1
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
python3 run_xgboostgpu_models.py --output_fp $OF"all_fold_metrics.csv" --drug_class 'targeted_other'  --use_mfp --use_dna --use_bc

