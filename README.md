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
python main.py --task_name cifar10 --cuda_id 0 --num_clients 4 --method moo --wandb_name rvfl_cifar10_s0 --seed 0
```

Note that, in the code "moo" and "rvfl" refer to our LASER-VFL method.
Local, Standard VFL, and Combinatorial are all special instances of
"decoupled" with and appropriate "--sets_of_clients_in_tasks" argument.

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
