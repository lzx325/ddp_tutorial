{
	source setup.sh

	mkdir -p slurm

	sbatch <<- EOF
	#!/bin/bash
	#SBATCH --nodes 1
	#SBATCH --ntasks-per-node 1
	#SBATCH -J torchrun
	#SBATCH --partition=preempt
	#SBATCH -o slurm/%J.out
	#SBATCH -e slurm/%J.err
	#SBATCH --time=1:00:00
	#SBATCH --cpus-per-task=4
	#SBATCH --gres=gpu:2
	
	export master_addr="\$(scontrol show hostname \${SLURM_NODELIST} | head -n 1)"
	echo "master_addr=\${master_addr}"
	export master_port=29400
	export LOGLEVEL="DEBUG"
	
	# using 2 nodes each with 8 GPUs
	srun --nodes=2 --ntasks-per-node=1 \
	python -u -m torch.distributed.run \
	--rdzv_conf="join_timeout=60,timeout=30" \
	--nproc_per_node=8 \
	--nnodes=2 \
	--rdzv_id=200 \
	--rdzv_backend=c10d \
	--rdzv_endpoint=\${master_addr}:\${master_port} \
	elastic_ddp.py

	EOF

	exit 0;
}