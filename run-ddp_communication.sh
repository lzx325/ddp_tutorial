{
	set -e

	module purge
	source "/home/liz0f/anaconda3/etc/profile.d/conda.sh"
	conda deactivate
	conda activate diffusers
	export LD_LIBRARY_PATH="/home/liz0f/anaconda3/envs/diffusers/lib:$LD_LIBRARY_PATH"

	mkdir -p slurm

	sbatch <<- EOF
	#!/bin/bash
	#SBATCH --nodes 2
	#SBATCH --ntasks-per-node 1
	#SBATCH -J torchrun
	#SBATCH --partition=batch
	#SBATCH -o slurm/%J.out
	#SBATCH -e slurm/%J.err
	#SBATCH --time=1:00:00
	#SBATCH --cpus-per-task=16
	#SBATCH --gres=gpu:8

	export master_addr="\$(scontrol show hostname \${SLURM_NODELIST} | head -n 1)"
	echo "master_addr=\${master_addr}"
	export master_port=29400
	export LOGLEVEL="DEBUG"
	
	# using 1 node with 8 GPUs
	if false; then
		srun --nodes=1 --ntasks-per-node=1 \
		python -u -m torch.distributed.run \
		--rdzv_conf="join_timeout=60,timeout=30" \
		--nnodes=1 \
		--nproc_per_node=8 \
		--rdzv_id=200 \
		--rdzv_backend=c10d \
		--rdzv_endpoint=\${master_addr}:\${master_port} \
		ddp_communication.py
	fi

	# another way of using 1 node with 8 GPUs (the --standalone option of torchrun)
	if false; then
		srun --nodes=1 --ntasks-per-node=1 \
		python -u -m torch.distributed.run \
		--rdzv_conf="join_timeout=60,timeout=30" \
		--standalone \
		--nproc_per_node=8 \
		ddp_communication.py
	fi

	# using 2 nodes each with 8 GPUs
	if false; then
		srun --nodes=2 --ntasks-per-node=1 \
		python -u -m torch.distributed.run \
		--rdzv_conf="join_timeout=60,timeout=30" \
		--nnodes=2 \
		--nproc_per_node=8 \
		--rdzv_id=200 \
		--rdzv_backend=c10d \
		--rdzv_endpoint=\${master_addr}:\${master_port} \
		ddp_communication.py
	fi
	EOF
}
	