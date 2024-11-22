#!/bin/bash
#SBATCH --job-name="luscas"
#SBATCH --mail-user=lucas.pacheco@unibe.ch
#SBATCH --mail-type=all
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=300G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --cpus-per-task=20

 module load GCCcore/12.3.0
 module load CMake/3.26.3-GCCcore-12.3.0
 module load Ninja/1.11.1-GCCcore-12.3.0
#  module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Python/3.11.3-GCCcore-12.3.0
 export CXX=/software.9/software/GCCcore/12.3.0/bin/g++
 export CC=/software.9/software/GCCcore/12.3.0/bin/gcc

rm -rfv *xml *json simulations

 #alias for requesting a server for processing in in teractive mode
 alias node_start="salloc --nodes=1 --ntasks-per-node=4 --mem-per-cpu=2G --time=5:00:00 --partition=gpu --gpus-per-node=rtx3090:1"

 bash ~/fl-ns3/run_sim.sh
