#!/usr/bin/env bash

# Usage:
#   ./scripts/tables/run_cifar10.sh [cuda_id]

CUDA_ID="${1:-0}"  # Default to 0 if no argument is given

python main.py --task_name cifar10 --method local --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method local --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method local --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method svfl --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method svfl --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method svfl --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method ensemble --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method ensemble --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method ensemble --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method combinatorial --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method plug --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method plug --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name cifar10 --method laser --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name cifar10 --method laser --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
