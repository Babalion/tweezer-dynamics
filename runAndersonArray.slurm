#!/bin/bash
# This script is used to start the simulation 'mpsPhonons.py' on the JUSTUS 2 HPC cluster
# It is based on the SLURM job scheduler

# Number of nodes to allocate, always 1
#SBATCH --nodes=1
# Number of MPI instances (ranks) to be executed per node, always 1
#SBATCH --ntasks-per-node=1
# Number of threads per MPI instance, dependin on the size of the MPS, you can change the value between 2 and 48
# for more than 70 atoms and phonon truncation >2 you can set already 48
#SBATCH --cpus-per-task=48
# Allocate xx GB memory per node, usually 20GB is more than enough
#SBATCH --mem=20gb
# Maximum run time of job, depends on accuracy, bond-dimension, size, difficult to guess. You have to test
#SBATCH --time=4-00:00:00
# Configure array parameters, split job in parts labeled 0-x. (only one job x=0)
#SBATCH --array 0-14
# Give job a reasonable name
#SBATCH --job-name=r0_9_l_k

# File name for standard output
# (%A will be replaced by the value of SLURM_ARRAY_JOB_ID and %a will be replaced by the value of SLURM_ARRAY_TASK_ID)
#SBATCH --output=r0_9_l_k-%A_%a.out
# File name for error output
#SBATCH --error=r0_9_l_k-%A_%a.err

# you can check the current log of all jobs with the command
# tail -fn 10 [FILENAME]-*
# you can check the job status with
# squeue -i 10

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export HOME=~

module load compiler/intel/19.1.2
module load mpi/impi
module load devel/valgrind
module load numlib/mkl/2020.2
srun $(ws_find conda)/conda/envs/quimbPet/bin/python ~/Anderson-localization/mpsPhonons.py ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT}
