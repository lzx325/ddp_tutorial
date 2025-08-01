{
	source setup.sh

	if false; then
		python -u -m torch.distributed.run \
		--standalone \
		--nproc_per_node=8 \
		torchrun_demo.py \
		1000 10
	fi

	if false; then
		LOGLEVEL="DEBUG" python -u -m torch.distributed.run \
		--nproc_per_node="$1" \
		--nnodes=2 \
		--rdzv_id=456 \
		--rdzv_backend=c10d \
		--rdzv_endpoint=gpu510-07:29603 \
		--rdzv_conf "join_timeout=60,timeout=30" \
		torchrun_demo.py \
		1000 10
	fi

	if true; then
	rm -f snapshot.pt
	mkdir -p slurm

	n_nodes=2
	n_gpus_per_node=4
	n_proc_per_gpu=3
	

	sbatch <<- EOF
	#!/bin/bash
	#SBATCH --account $ACCOUNT
	#SBATCH --nodes $n_nodes
	#SBATCH --ntasks-per-node 1
	#SBATCH -J run-torchrun
	#SBATCH --partition=preempt
	#SBATCH -o slurm/%J.out
	#SBATCH -e slurm/%J.err
	#SBATCH --time=1:00:00
	#SBATCH --cpus-per-task=$((n_gpus_per_node*n_proc_per_gpu))
	#SBATCH --gres=gpu:$n_gpus_per_node
	
	export master_addr="\$(scontrol show hostname \${SLURM_NODELIST} | head -n 1)"
	echo "master_addr=\${master_addr}"
	export master_port=29400
	export LOGLEVEL="DEBUG"

	# using 2 nodes each with 8 GPUs
	srun --nodes=$n_nodes --ntasks-per-node=1 \
	python -u -m torch.distributed.run \
	--rdzv_conf="join_timeout=60,timeout=30" \
	--nproc_per_node=$n_gpus_per_node \
	--nnodes=$n_nodes \
	--rdzv_id=200 \
	--rdzv_backend=c10d \
	--rdzv_endpoint=\${master_addr}:\${master_port} \
	torchrun_demo.py \
	1000 10

	EOF
	fi

	exit 0;
}