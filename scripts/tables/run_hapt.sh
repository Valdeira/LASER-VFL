#!/usr/bin/env bash

# Usage:
#   ./scripts/tables/run_hapt.sh [cuda_id]

CUDA_ID="${1:-0}"  # Default to 0 if no argument is given

python main.py --task_name hapt --method local --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method local --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method local --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name hapt --method svfl --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method svfl --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method svfl --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name hapt --method ensemble --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method ensemble --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method ensemble --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name hapt --method combinatorial --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method combinatorial --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method combinatorial --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name hapt --method plug --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method plug --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method plug --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"

python main.py --task_name hapt --method laser --p_miss_train 0.0 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method laser --p_miss_train 0.1 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
python main.py --task_name hapt --method laser --p_miss_train 0.5 --seed 0 1 2 3 4 --cuda_id "${CUDA_ID}"
