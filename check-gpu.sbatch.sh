#!/bin/bash
#SBATCH --job-name="luscas"
#SBATCH --mail-user=lucas.pacheco@unibe.ch
#SBATCH --mail-type=all
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:2
#SBATCH --cpus-per-task=10

 module load GCCcore/12.3.0
 module load CMake/3.26.3-GCCcore-12.3.0
 module load Ninja/1.11.1-GCCcore-12.3.0
#  module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Python/3.11.3-GCCcore-12.3.0
 export CXX=/software.9/software/GCCcore/12.3.0/bin/g++
 export CC=/software.9/software/GCCcore/12.3.0/bin/gcc

python scratch/server.py
