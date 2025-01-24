# LASER-VFL

This repository contains the code used for the experiments in the paper:  
**[Vertical Federated Learning with Missing Features During Training and Inference](https://arxiv.org/abs/2410.22564)**.

Pedro Valdeira, Shiqiang Wang, Yuejie Chi.

### Abstract

Vertical federated learning trains models from feature-partitioned datasets across multiple clients, who collaborate without sharing their local data. Standard approaches assume that all feature partitions are available during both training and inference. Yet, in practice, this assumption rarely holds, as for many samples only a subset of the clients observe their partition. However, not utilizing incomplete samples during training harms generalization, and not supporting them during inference limits the utility of the model. Moreover, if any client leaves the federation after training, its partition becomes unavailable, rendering the learned model unusable. Missing feature blocks are therefore a key challenge limiting the applicability of vertical federated learning in real-world scenarios. To address this, we propose LASER-VFL, a vertical federated learning method for efficient training and inference of split neural network-based models that is capable of handling arbitrary sets of partitions. Our approach is simple yet effective, relying on the sharing of model parameters and on task-sampling to train a family of predictors. We show that LASER-VFL achieves a $\mathcal{O}({1}/{\sqrt{T}})$ convergence rate for nonconvex objectives and, under the Polyak-Łojasiewicz inequality, it achieves linear convergence to a neighborhood of the optimum. Numerical experiments show improved performance of LASER-VFL over the baselines. Remarkably, this is the case even in the absence of missing features. For example, for CIFAR-100, we see an improvement in accuracy of $18.2\%$ when each of four feature blocks is observed with a probability of 0.5 and of $7.4\%$ when all features are observed.

### Usage

To set up the environment, run the following command:

```bash
conda env create -f environment.yaml
```

Next, activate the environment:

```bash
conda activate vfl
```

To run an experiment, simply run main.py with the appropriate arguments. For example:

```bash
python main.py --task_name cifar10 --cuda_id 0 --num_clients 4 --method laser --seed 0
```

To get the results for Table 1, run the following command with the appropriate arguments. For example:

```bash
python get_final_metrics.py --project_name WANDB_PROJECT
```

Replace WANDB_PROJECT with the appropriate configuration, such as [wandb-username]/[wandb-project].

To get the scalability results, run the following command with the appropriate arguments. For example:

```bash
python results/plot_scalability.py --project_name WANDB_PROJECT
```

As above, replace WANDB_PROJECT with the appropriate configuration, such as [wandb-username]/[wandb-project].

### Citation

If you find this repository useful, please cite [our work](https://arxiv.org/abs/2410.22564):

```bibtex
@article{valdeira2024vertical,
  title={Vertical Federated Learning with Missing Features During Training and Inference},
  author={Valdeira, Pedro and Wang, Shiqiang and Chi, Yuejie},
  journal={arXiv preprint arXiv:2410.22564},
  year={2024}
}
```
