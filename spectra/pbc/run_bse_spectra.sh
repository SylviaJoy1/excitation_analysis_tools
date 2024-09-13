#!/bin/bash
#SBATCH --account=berkelbach
#SBATCH --job-name=6SiBSEspectrum
#SBATCH --time=11:40:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=190000MB ##700000MB 
#SBATCH --array=1

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=sjb2225@columbia.edu

JOBNAME=$SLURM_JOB_NAME

. /burg/berkelbach/users/sjb2225/build/spack/share/spack/setup-env.sh
spack env activate -p pyscf
export PYTHONPATH=/burg/berkelbach/users/sjb2225/build/pyscf:$PYTHONPATH
export PYSCF_MAX_MEMORY=180000 #690000
export OMP_NUM_THREADS=32

kshift=$1
basis=$2
nk=$3
sc=$4
python run_bse_spectra.py ${SLURM_ARRAY_TASK_ID} ${1} ${2} ${3} ${4} > ${1}_${2}_${3}_${4}_${SLURM_ARRAY_TASK_ID}.txt
