import torch

from utils import time_decorator
from methods.method_utils import get_f1


@time_decorator
def test_plug(dataloader, models, criterion, device, is_final=False, compute_f1=False, is_train_data=False):
    
    for model in models:
        model.eval()
    
    num_models = len(models)
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l, correct_l = [0.0] * num_models, [0.0] * num_models
    if compute_f1:
        true_positive_l, false_positive_l, false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models
    if compute_f1 and is_final:
        final_true_positive_l, final_false_positive_l, final_false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models

    final_correct = 0.0
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)

            # if client zero (active party) unobserved, skip batch
            if not mask[-1].item():
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue

            for i, model in enumerate(models):
                outputs = model(inputs, mask)
                loss = criterion(outputs, targets)
                loss_l[i] += loss.item()
                predicted = outputs.argmax(1)
                correct_l[i] += (predicted == targets).float().sum().item()
                if compute_f1:
                    true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                if is_final:
                    final_correct += (predicted == targets).float().sum().item()
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
            final_f1_per_predictor = [f1 for f1 in final_f1_per_predictor if f1 != 0.0] # remove models that were not used for prediction (do this more cleanly later using a dictionary)
            metrics[f"final_{data_split_type}_f1"] = sum(final_f1_per_predictor) / len(final_f1_per_predictor)
    else:
        avg_loss = [loss / num_batches for loss in loss_l]
        accuracy = [100 * correct / num_samples for correct in correct_l]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

@time_decorator
def train_plug(dataloader, models, optimizers, criterion, args, compute_f1=False):
    
    for model in models:
        model.train()
    
    num_models = len(models)
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l, correct_l = [0.0] * num_models, [0.0] * num_models
    if compute_f1:
        true_positive_l, false_positive_l, false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models
    num_samples_per_block_l = [num_samples] * num_models
    num_batches_per_block_l = [num_batches] * num_models

    for batch_num, batch in enumerate(dataloader):
        *inputs, targets, mask = batch
        inputs, targets = [tensor.to(args.device) for tensor in inputs], targets.to(args.device)

        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            
            # if client zero (active party) unobserved, skip batch
            if not mask[-1].item():
                num_batches_per_block_l[i] -= 1
                num_samples_per_block_l[i] -= len(inputs)
                continue

            optimizer.zero_grad()
            outputs = model(inputs, mask, args.p_drop)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_l[i] += loss.item()
            predicted = outputs.argmax(1)
            correct_l[i] += (predicted == targets).float().sum().item()
            if compute_f1:
                true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
        
        if (batch_num + 1) % 25 == 0:
            try:
                loss_value = loss.item()
                print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): {loss_value:.4f}")
            except UnboundLocalError:
                print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): N/A")
            
            
    avg_loss = [loss / num_batches_per_block_l[i] for i, loss in enumerate(loss_l)]
    accuracy = [100 * correct / num_samples_per_block_l[i] for i, correct in enumerate(correct_l)]
    metrics = {"train_loss": avg_loss, "train_acc": accuracy}
    if compute_f1:
        metrics["train_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics
