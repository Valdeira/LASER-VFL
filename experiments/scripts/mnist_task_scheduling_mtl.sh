# with wandb: ./experiments/scripts/mnist_task_scheduling_mtl.sh 3 0
# without wandb: ./experiments/scripts/mnist_task_scheduling_mtl.sh 3 0 false
CUDA_ID=$1
SEED=$2
USE_WANDB=${3:-true} # default to true if no 3rd argument is given

# omitted the following arguments:
# scheduler_milestones
# scheduler_gamma missing
# aggregation
# clients_in_model

# Base command setup
COMMAND="python experiments/train.py \
    --cuda_id $CUDA_ID \
    --seed $SEED \
    --n_epochs 30 \
    --lr 1.0 \
    --num_clients 4 \
    --batch_size 1024 \
    --weight_decay 0.0 \
    --momentum 0.0 \
    --dataset mnist \
    --architecture shallow \
    --method task_scheduling_mtl"

if [ "$USE_WANDB" = "false" ]; then
    COMMAND="$COMMAND --no_wandb"
fi

$COMMAND
