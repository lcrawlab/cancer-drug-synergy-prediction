#!/bin/bash

#SBATCH --mail-type=ALL                                     #email notification
#SBATCH --mail-user=alexandra_wong@brown.edu                #email address
#SBATCH -J 20251002_combo_mfp_pcnngl_csreg_experiment      #job name

#SBATCH --partition=gpu                                     #gpu partition
#SBATCH --gres=gpu:1                                        #number of gpus
#SBATCH -N 1                                                #number of nodes
#SBATCH -c 4                                                #number of cores

#SBATCH -t 12:00:00                                         #time limit (HH:MM:SS)
#SBATCH --mem=10G                                          #memory per node

#SBATCH --array=1                                         #job array

# Use '%A' for job array ID, '%J' for job ID, '%a' for array task ID
#SBATCH --output=scratch/anticancer-synergy-prediction-scratch/experiments/20251002_combo_pcnngl/mfp_pcnngl_csreg_%a_slurm.out           #output file
#SBATCH --error=scratch/anticancer-synergy-prediction-scratch/experiments/20251002_combo_pcnngl/mfp_pcnngl_csreg_%a_slurm.err            #error file

source env/bin/activate

SCRATCH="scratch/anticancer-synergy-prediction-scratch/"
export PYTHONPATH="${PYTHONPATH}:~/cancer-drug-synergy-prediction/"
PYOF=$SCRATCH"experiments/20251002_combo_pcnngl/mfp_pcnngl_csreg"$SLURM_ARRAY_TASK_ID".txt"
PLOT=$SCRATCH"experiments/20251002_combo_pcnngl/csreg/mfp/"
mkdir -p $PLOT
# Run hyperparameter experiments using job array index to select hyperparameters in the vnn_experiments_oscar.py script
# REMEMBER TO INDEX BY 1
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME starting on `date`"
python3 models/src/modelPCNNGL.py --score 'COMBOSCORE' --use_mfp --use_csreg --output_fp $PLOT > $PYOF
echo "Job $SLURM_ARRAY_TASK_ID ended at `date`"