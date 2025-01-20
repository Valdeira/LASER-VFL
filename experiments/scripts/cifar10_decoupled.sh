# with wandb: ./experiments/scripts/cifar10_decoupled.sh 1 0 true
# without wandb: ./experiments/scripts/cifar10_decoupled_powerset.sh 1 0 false
#!/bin/bash
CUDA_ID=$1
SEED=$2
USE_WANDB=${3:-true}  # default to true if no 3rd argument is given

# omitted the following arguments:
# --scheduler_milestones
# --scheduler_gamma missing
# --name
# --clients_in_model

COMMAND="python experiments/train_decoupled.py \
    --cuda_id $CUDA_ID \
    --seed $SEED \
    --n_epochs 100 \
    --lr 0.2 \
    --num_clients 4 \
    --batch_size 1024 \
    --weight_decay 0.001 \
    --momentum 0.5 \
    --dataset cifar10 \
    --architecture resnet18 \
    --aggregation mean
    "

if [ "$USE_WANDB" = "false" ]; then
    COMMAND="$COMMAND --no_wandb"
fi

$COMMAND
