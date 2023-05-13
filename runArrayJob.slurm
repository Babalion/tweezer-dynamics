#!/bin/bash
# Number of nodes to allocate
#SBATCH --nodes=1
# Number of MPI instances (ranks) to be executed per node
#SBATCH --ntasks-per-node=1
# Number of threads per MPI instance
#SBATCH --cpus-per-task=24
# Allocate xx GB memory per node
#SBATCH --mem=20gb
# Maximum run time of job
#SBATCH --time=0-12:00:00
# Configure array parameters
#SBATCH --array 0-2
# Give job a reasonable name
#SBATCH --job-name=FockState1_3

# File name for standard output
# (%A will be replaced by the value of SLURM_ARRAY_JOB_ID
# and %a will be replaced by the value of SLURM_ARRAY_TASK_ID)
#SBATCH --output=FockState1_3-%A_%a.out
# File name for error output
#SBATCH --error=FockState1_3-%A_%a.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HOME=~

module load compiler/intel/19.1.2
module load mpi/impi
module load devel/valgrind
module load numlib/mkl/2020.2
srun $(ws_find conda)/conda/envs/quimbPet/bin/python ~/Anderson-localization/mpsPhonons.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT}
