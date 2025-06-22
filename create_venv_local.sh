#!/bin/bash

# install, create, and activate virtualenv called "env"
python3.11 -m pip install virtualenv #Change this to python3 on oscar
python3.11 -m venv env #Change this to python3 on oscar
module load cuda/12.1.1 # - needed for running on gpus
module load cudnn/8.7.0.84-11.8 # - needed for running on gpus
source env/bin/activate

# update pip and install required packages
pip install -U pip
pip install -r requirements.txt

echo created venv
