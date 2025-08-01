CONDA_SH="/cm/shared/apps/Mambaforge/24.3.0-0/etc/profile.d/conda.sh"
CONDA_ENV_PREFIX="/scratch/m000071-pm02/lzx325/env/SP"
ACCOUNT="marlowe-m000071"

set -e
module purge
module load slurm

source "$CONDA_SH"
conda deactivate
conda activate "$CONDA_ENV_PREFIX"
export LD_LIBRARY_PATH="$CONDA_ENV_PREFIX/lib:$LD_LIBRARY_PATH"