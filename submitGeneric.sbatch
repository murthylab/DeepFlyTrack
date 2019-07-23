#!/bin/bash
#SBATCH --mem=40000
#SBATCH --time=12:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load anaconda
module load cudnn
module load cudann
module load cudatoolkit
python $1 $2
