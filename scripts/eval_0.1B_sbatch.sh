#!/bin/bash
#SBATCH --job-name=slurm-test # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --mail-type=ALL # send email when job begins, ends or failed etc. 

#SBATCH -o gpt-neox-%j.log
#SBATCH -e gpt-neox-%j.err

NNODES=1
GPUS_PER_NODE=1

CODE_ROOT='/cognitive_comp/yangping/nlp/gpt-neox'
CONFIG_ROOT='/cognitive_comp/yangping/nlp/gpt-neox/custom_config/01B'
MODEL_CONFIG="01B.yml"
# LOCAL_CONFIG="slurm_local.yml"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "

# 因为这里复用的代码是用deepy.py里面的，默认传入了一个script_path，我们这里没用到，所以实际上第二个参数是无效的，可以不用关注。
export RUN_CMD="$CODE_ROOT/deepy.py $CODE_ROOT/evaluate.py -d $CONFIG_ROOT $MODEL_CONFIG"

srun python $RUN_CMD