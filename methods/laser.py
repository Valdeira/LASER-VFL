import math

import torch

from utils import time_decorator
from methods.method_utils import get_f1


@time_decorator
def test_laser(dataloader, models, criterion, device, is_final=False, compute_f1=False, is_train_data=False):
    
    [model] = models

    model.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)

    loss_d = {clients_l: 0.0 for clients_l in model.powerset}
    correct_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1:
        true_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_negative_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1 and is_final:
        final_true_positive_l = [0.0 for _ in range(model.num_clients)]
        final_false_positive_l = [0.0 for _ in range(model.num_clients)]
        final_false_negative_l = [0.0 for _ in range(model.num_clients)]

    final_correct = 0.0 
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)
            
            if torch.sum(mask).item() == 0:
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue

            outputs_per_head_l = model(inputs, training=False)
            for i, outputs_per_task_d in enumerate(outputs_per_head_l):
                for clients_subset, outputs in outputs_per_task_d.items():
                    loss = criterion(outputs, targets)
                    
                    # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
                    # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
                    norm_constant = len(clients_subset)
                    loss_d[clients_subset] += loss.item() / norm_constant
                    predicted = outputs.argmax(1)
                    correct_d[clients_subset] += (predicted == targets).float().sum().item() / norm_constant
                    if compute_f1:
                        # this is not the f1 of any given predictor, but rather the f1 obtain from the average
                        # TP, FP, and FN across the different predictors for the same set of blocks (diff from final result)
                        true_positive_d[clients_subset] += ((predicted == 1) & (targets == 1)).sum().item() / norm_constant
                        false_positive_d[clients_subset] += ((predicted == 1) & (targets == 0)).sum().item() / norm_constant
                        false_negative_d[clients_subset] += ((predicted == 0) & (targets == 1)).sum().item() / norm_constant

                    # check if final (after training) and if this task uses the (exact) set of observed blocks
                    if is_final and set(clients_subset) == set(torch.nonzero(mask).view(-1).tolist()):
                        final_correct += (predicted == targets).float().sum().item() / len(clients_subset)
                        if compute_f1:
                            final_true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                            final_false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                            final_false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_split_type}_acc"] = 100 * final_correct / num_samples
        if compute_f1:
            final_f1_per_predictor = [get_f1(tp, fp, fn) for tp, fp, fn in zip(final_true_positive_l, final_false_positive_l, final_false_negative_l)]
            metrics[f"final_{data_split_type}_f1"] = sum(final_f1_per_predictor) / len(final_f1_per_predictor)
    else:
        avg_loss = [loss / num_batches for loss in [loss_d[clients_l] for clients_l in model.powerset]]
        accuracy = [100 * correct / num_samples for correct in [correct_d[clients_l] for clients_l in model.powerset]]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            true_positive = [true_positive_d[clients_l] for clients_l in model.powerset]
            false_positive = [false_positive_d[clients_l] for clients_l in model.powerset]
            false_negative = [false_negative_d[clients_l] for clients_l in model.powerset]
            avg_loss = [loss / num_batches for loss in [loss_d[clients_l] for clients_l in model.powerset]]
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive, false_positive, false_negative)]

    return metrics

@time_decorator
def train_laser(dataloader, models, optimizers, criterion, args, compute_f1=False):
    
    [model], [optimizer] = models, optimizers

    device = args.device

    model.train()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    loss_d = {clients_l: 0.0 for clients_l in model.powerset}
    correct_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1:
        true_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_negative_d = {clients_l: 0.0 for clients_l in model.powerset}
    
    for batch_num, batch in enumerate(dataloader):
        *inputs, targets, mask = batch
        inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)
        
        if torch.sum(mask).item() == 0:
            num_batches -= 1 # in practice, this batch is not used
            num_samples -= len(inputs[0]) # in practice, these samples are not used
            continue

        optimizer.zero_grad()

        observed_blocks = torch.nonzero(mask).view(-1).tolist()

        outputs_per_head_l = model(inputs, training=True, observed_blocks=observed_blocks)
        
        total_loss = 0
        for outputs_per_task_d in outputs_per_head_l:

            head_loss = 0
            for clients_subset, outputs in outputs_per_task_d.items():
                loss = criterion(outputs, targets)

                norm_constant = 1.0
                norm_constant = norm_constant / len(clients_subset) # account for multiple heads
                
                # if a given element (client) must be in a set, then we do n-1 choose k-1 instead
                n, k = (len(observed_blocks)-1, len(clients_subset)-1)
                norm_constant = norm_constant * math.comb(n, k)

                head_loss += loss * norm_constant

                # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
                # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
                loss_d[clients_subset] += loss.item() * norm_constant
                predicted = outputs.argmax(1)
                correct_d[clients_subset] += (predicted == targets).float().sum().item() * norm_constant
                if compute_f1:
                        # this is not the f1 of any given predictor, but rather the f1 obtain from the average
                        # TP, FP, and FN across the different predictors for the same set of blocks (diff from final result)
                        true_positive_d[clients_subset] += ((predicted == 1) & (targets == 1)).sum().item() / norm_constant
                        false_positive_d[clients_subset] += ((predicted == 1) & (targets == 0)).sum().item() / norm_constant
                        false_negative_d[clients_subset] += ((predicted == 0) & (targets == 1)).sum().item() / norm_constant

            total_loss += head_loss

        total_loss.backward()
        optimizer.step()
        
        if (batch_num + 1) % 25 == 0:
            print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): {loss.item():.4f}")
            
    loss_l = [loss_d[clients_l] for clients_l in model.powerset]
    correct_l = [correct_d[clients_l] for clients_l in model.powerset]

    avg_loss = [loss / num_batches for loss in loss_l]
    accuracy = [100 * correct / num_samples for correct in correct_l]
    metrics = {"train_loss": avg_loss, "train_acc": accuracy}
    if compute_f1:
        true_positive_l = [true_positive_d[clients_l] for clients_l in model.powerset]
        false_positive_l = [false_positive_d[clients_l] for clients_l in model.powerset]
        false_negative_l = [false_negative_d[clients_l] for clients_l in model.powerset]
        metrics["train_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics
