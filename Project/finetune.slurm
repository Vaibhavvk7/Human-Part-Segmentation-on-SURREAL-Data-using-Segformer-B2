#!/bin/bash
#SBATCH --job-name=finetune_segformer
#SBATCH --output=logs_finetune/finetune_%j.out
#SBATCH --error=logs_finetune/finetune_%j.err
#SBATCH --partition=gpu-interactive        # or courses-gpu if that’s your main GPU queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# ===== Conda environment =====
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate surreal_arm

# ===== Environment tuning =====
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Optional: safer PyTorch multiprocessing on clusters
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== Paths =====
DATA_ROOT=/home/channagiri.b/SmallData_Project/Dataset/SURREAL/data
OUT_DIR=/home/channagiri.b/SmallData_Project/Output_FineTune
mkdir -p "${OUT_DIR}" logs_finetune

# ===== Run Fine-tuning =====
echo "=== Starting fine-tune job on node: $(hostname) ==="
nvidia-smi

python finetuning.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --batch_size 8 \
  --num_workers 0 \
  --img_size 160 160
