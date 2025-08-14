#!/bin/bash
#SBATCH --job-name=bridge_noAugSteps_DeltePauseFrame_deepspeed_bs16gpus32gas1_tuneProjectionDitHead   # name
#SBATCH -p efm_p
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/logs/0807/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/logs/0807/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-49
# ---------- 通信环境 ----------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))


cd /mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1


export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
echo "Total GPUs: $TOTAL_GPUS"

export run_root_dir=/mnt/petrelfs/wangfangjing/code/Isaac-GR00T-M1/playground/Checkpoints
export run_id=0807_bridge_noAugSteps_DeltePauseFrame_tuneProjectionDitHead_bs16gpus32gas1

output_dir=${run_root_dir}/${run_id}
export output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


# ---------- 启动 ----------
srun torchrun \
  --nnodes $SLURM_NNODES \
  --nproc_per_node 8 \
  --rdzv_backend c10d \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  scripts/gr00t_finetune_multinode.py \
  --dataset_path playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot \
  --run_name ${run_id} \
  --output_dir ${output_dir} \
  --data_config oxe_bridge \
  --num_gpus 8 \
  --batch_size 16 \
  --delte_pause_frame \
  --augsteps 0 \
  --save_steps 3000 \
  --max_steps 65000 \
  --deepspeed_config scripts/PCluster/deepspeed_zero2.json

