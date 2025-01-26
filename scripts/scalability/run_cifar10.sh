#!/usr/bin/env bash

# Usage:
#   ./scripts/scalability/run_cifar10.sh [cuda_id]

CUDA_ID="${1:-0}"  # Default to 0 if no argument is given

# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 2 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 3 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 4 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 5 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 6 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 7 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method local --p_miss_train 0.1 --num_clients 8 --seed 0 1 2 --cuda_id "${CUDA_ID}"

# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 2 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 3 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 4 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 5 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 6 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 7 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --num_clients 8 --seed 0 1 2 --cuda_id "${CUDA_ID}"

# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 2 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 3 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 4 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 5 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 6 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 7 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --num_clients 8 --seed 0 1 2 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --num_clients 2 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --num_clients 3 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --num_clients 4 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --num_clients 5 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --num_clients 6 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --num_clients 7 --seed 0 1 2 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 2 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 3 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 4 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 5 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 6 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 7 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --num_clients 8 --seed 0 1 2 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 2 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 3 --seed 0 1 2 --cuda_id "${CUDA_ID}"
# python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 4 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 5 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 6 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 7 --seed 0 1 2 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --num_clients 8 --seed 0 1 2 --cuda_id "${CUDA_ID}"
