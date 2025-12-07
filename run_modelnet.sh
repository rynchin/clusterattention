#!/bin/bash
#SBATCH -p iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=0-23:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=32GB
#SBATCH -o logs/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/%j.err  # File to which STDERR will be written, %j inserts jobid

# Create logs directory if it doesn't exist
mkdir -p logs

echo "$@"

source ~/clusterattention/venv/bin/activate
export MATPLOTLIBRC=$HOME/.matplotlib/matplotlibrc

module load cuda/12.4.1-fasrc01
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export TF_GPU_ALLOCATOR=cuda_malloc_async

# run code
export NUMBA_DISABLE_JIT=1
export HYDRA_FULL_ERROR=1
cd ~/clusterattention

echo "Running train_modelnet.py"
python train_modelnet.py --runs modelnet