#!/bin/bash
#SBATCH --job-name=bridge_noAugSteps_DeltePauseFrame   # name
#SBATCH -p efm_p
#SBATCH -N 1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/logs/0804/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/logs/0804/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-49

# [8,34,47,49,93-94]
# SH-IDCA1404-10-140-54-25 

# source ~/.bashrc     # 确保 conda 命令可用
# source ~/.zshrc
# source ~/envs4jinhui.sh
# proxy_on

# conda activate llavavla310  # 替换为你的环境名

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))



cd /mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1
# conda activate llavavla310
# proxy_on


export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
export global_batch_size=$((TOTAL_GPUS * vla_per_device_batch_size)) # 512 is the default global batch size, you can change it if needed
echo "Total GPUs: $TOTAL_GPUS"
export run_root_dir=/mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/Checkpoints
export run_id=0804_bridge_noAugSteps_DeltePauseFrame

output_dir=${run_root_dir}/${run_id}
export output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/



srun --jobid $SLURM_JOBID bash -c 'python scripts/gr00t_finetune.py \
--dataset_path /mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot \
--run_name ${run_id} \
--output_dir ${output_dir} \
--data_config oxe_bridge \
--batch_size 32 \
--max_steps 100000 \
--num_gpus 8 \
--delte_pause_frame \
--augsteps 0 \
--save_steps 10000'
