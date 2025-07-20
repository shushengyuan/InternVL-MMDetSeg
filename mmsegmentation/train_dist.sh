#!/bin/bash

# 分布式训练脚本 - 2GPU配置，解决NCCL和多GPU问题
set -e  # 遇到错误立即退出

echo "=== 开始分布式训练 DNANet (2 GPU) ==="

# 0. 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate internvl-mmdetseg
echo "已激活环境: $(conda info --envs | grep '*')"

# 1. 设置NCCL环境变量 - 解决通信问题
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1  
export NCCL_SOCKET_IFNAME=lo  # 本地回环，适用于单机多卡
export NCCL_DEBUG=WARN  # 减少日志输出，只显示警告
export CUDA_LAUNCH_BLOCKING=0  # 允许异步CUDA操作
export NCCL_TIMEOUT=1800  # 30分钟超时

# 2. 设置PyTorch分布式环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29501  # 避免端口冲突

# 3. 设置OMP线程数，避免CPU争用
export OMP_NUM_THREADS=8

# 4. GPU配置 - 使用GPU 1,2 (避开被占用的GPU 0)
# export CUDA_VISIBLE_DEVICES=1,2
GPUS=2  # 使用2张GPU
CONFIG="configs/dnanet/upernet.py"

echo "使用GPU: 1,2 (避开占用的GPU 0)"
echo "GPU数量: $GPUS"
echo "配置文件: $CONFIG"
echo "NCCL环境变量已设置"

# 5. 检查GPU状态
echo "=== 当前GPU状态 ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

# 6. 清理可能的僵尸进程（但不影响正在运行的训练）
echo "=== 检查端口占用 ==="
lsof -i :29501 || echo "端口29501空闲"

# 7. 启动分布式训练
echo "=== 启动2GPU分布式训练 ==="
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=29501 \
    tools/train.py $CONFIG \
    --launcher pytorch \
    --seed 42

echo "=== 训练完成 ===" 