#!/bin/bash

#SBATCH --mail-type=ALL                                     #email notification
#SBATCH --mail-user=alexandra_wong@brown.edu                #email address
#SBATCH -J 20240718_modelPCNNGL_hp_mfpprot_bc0                    #job name

#SBATCH --partition=gpu                                     #gpu partition
#SBATCH --gres=gpu:1                                        #number of gpus
#SBATCH -N 1                                                #number of nodes
#SBATCH -c 4                                                #number of cores

#SBATCH -t 5:00:00                                         #time limit (HH:MM:SS)
#SBATCH --mem=5G                                           #memory per node

#SBATCH --array=1                                        #job array

# Use '%A' for job array ID, '%J' for job ID, '%a' for array task ID
#SBATCH --output=scratch/anticancer-synergy-prediction-scratch/experiments/20240718_modelPCNNGL/bc0/mfpprot/slurm_%a.out           #output file
#SBATCH --error=scratch/anticancer-synergy-prediction-scratch/experiments/20240718_modelPCNNGL/bc0/mfpprot/slurm_%a.err            #error file


source env/bin/activate

SCRATCH="scratch/anticancer-synergy-prediction-scratch/"

OF=$SCRATCH"experiments/20240718_modelPCNNGL/bc0/mfpprot/"

# Run hyperparameter experiments using job array index to select hp_mfpprot_bc0 in the vnn_experiments_oscar.py script
# REMEMBER TO INDEX BY 1
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"
python3 modelPCNNGL.py --output_fp $OF --job_index $SLURM_ARRAY_TASK_ID --use_mfp --use_prot --use_bc

