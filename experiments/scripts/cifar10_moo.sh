# with wandb: ./experiments/scripts/cifar10_moo.sh 0 0 true true
# without wandb: ./experiments/scripts/cifar10_moo.sh 0 0 false true
#!/bin/bash
CUDA_ID=$1
SEED=$2
USE_WANDB=${3:-true}  # default to true if no 3rd argument is given
IS_L2_LOSS=${4:-false}  # default to false if no 4th argument is given
IS_SINGLE_HEAD=${5:-false}  # default to false if no 5th argument is given

# omitted the following arguments:
# --scheduler_milestones
# --scheduler_gamma missing
# --name

COMMAND="python experiments/train_moo.py \
    --cuda_id $CUDA_ID \
    --seed $SEED \
    --n_epochs 100 \
    --lr 0.1 \
    --num_clients 4 \
    --batch_size 1024 \
    --weight_decay 0.008 \
    --momentum 0.5 \
    --dataset cifar10 \
    --architecture resnet18 \
    --aggregation mean \
    --is_sampled_aggregation
    "

if [ "$USE_WANDB" = "false" ]; then
    COMMAND="$COMMAND --no_wandb"
fi

if [ "$IS_L2_LOSS" = "true" ]; then
    COMMAND="$COMMAND --l2_loss"
fi

if [ "$IS_SINGLE_HEAD" = "true" ]; then
    COMMAND="$COMMAND --is_single_head"
fi

$COMMAND
