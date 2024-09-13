#!/bin/bash
#SBATCH --account=berkelbach
#SBATCH --job-name=plot_parallel_CIS_spectra
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64000MB ##700000MB 
###SBATCH --array=

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=sjb2225@columbia.edu

JOBNAME=$SLURM_JOB_NAME

. /burg/berkelbach/users/sjb2225/build/spack/share/spack/setup-env.sh
spack env activate -p pyscf
export PYTHONPATH=/burg/berkelbach/users/sjb2225/build/pyscf:$PYTHONPATH
export PYSCF_MAX_MEMORY=180000 #690000
export OMP_NUM_THREADS=1

kshift=$1
basis=$2
nk=$3
trans_corr=$4
python plot_parallel_bse_spectrum.py  ${1} ${2} ${3} ${4} > ${1}_${2}_${3}_${4}_LDA.txt
