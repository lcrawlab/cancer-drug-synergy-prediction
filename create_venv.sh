#!/bin/bash

# install, create, and activate virtualenv called "env"
python3.11 -m pip install virtualenv
python3.11 -m venv env
module load cuda/12.1.1 # - needed for running on gpus
module load cudnn/8.7.0.84-11.8 # - needed for running on gpus
source env/bin/activate

# update pip and install required packages
pip install -U pip
pip install -r requirements.txt

echo created venv
