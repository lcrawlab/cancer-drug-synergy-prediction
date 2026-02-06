#!/bin/bash

# install, create, and activate virtualenv called "env"
python3 -m pip install virtualenv
python3 -m venv env
module load cuda/12.6.0 # - needed for running on gpus
module load cudnn/9.8.0.87-12 # - needed for running on gpus
source env/bin/activate

# update pip and install required packages
pip install -U pip
pip install -r requirements.txt

echo created venv
