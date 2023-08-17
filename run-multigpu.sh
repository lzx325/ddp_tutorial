{
	set -e

	module purge
	source "/home/liz0f/anaconda3/etc/profile.d/conda.sh"
	conda deactivate
	conda activate diffusers
	export LD_LIBRARY_PATH="/home/liz0f/anaconda3/envs/diffusers/lib:$LD_LIBRARY_PATH"

	# example of launching each individual worker process manually without using any launcher
	if false; then
		for i in $(seq "$1" "$2"); do
			python -u multigpu.py 4 "$i" $(($i%8)) --master_addr gpu210-06 --master_port 22355  > "$(hostname).$i.log" 2>&1 &
		done
	fi

	# using srun to run each worker
	if false; then
		sbatch <<- EOF
		#!/bin/bash
		#SBATCH --nodes 1
		#SBATCH --ntasks 8
		#SBATCH -J multigpu
		#SBATCH --partition=batch
		#SBATCH -o slurm/%J.out
		#SBATCH -e slurm/%J.err
		#SBATCH --time=1:00:00
		#SBATCH --cpus-per-task=2
		#SBATCH --gres=gpu:8
		for i in \$(seq 0 7); do
			srun -u --ntasks=1 \
			python -u multigpu.py 8 "\$i" &
		done
		wait
		EOF
	fi

	

	exit 0;
}