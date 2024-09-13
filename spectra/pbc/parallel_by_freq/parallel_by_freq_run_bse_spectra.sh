#!/bin/bash
#SBATCH --account=berkelbach
#SBATCH --job-name=shifted_8SiBSEspectrum
#SBATCH --time=11:40:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=690000MB ##700000MB 
#SBATCH --array=1

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=sjb2225@columbia.edu

JOBNAME=$SLURM_JOB_NAME

. /burg/berkelbach/users/sjb2225/build/spack/share/spack/setup-env.sh
spack env activate -p pyscf
export PYTHONPATH=/burg/berkelbach/users/sjb2225/build/pyscf:$PYTHONPATH
export PYSCF_TMPDIR=/burg/berkelbach/users/sjb2225/v2.4.0/bse_master/si/spectra
export PYSCF_MAX_MEMORY=680000 #690000
export OMP_NUM_THREADS=32

kshift=$1
basis=$2
nk=$3
dipole_corr=$4
python parallel_by_freq_run_bse_spectra.py ${SLURM_ARRAY_TASK_ID} ${1} ${2} ${3} ${4} > ${1}_${2}_${3}_${4}_${SLURM_ARRAY_TASK_ID}.txt
