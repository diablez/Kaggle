from __future__ import annotations

from collections.abc import Iterable

import torch


def run_epoch(model, loader: Iterable, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    predictions, targets = [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (batch_predictions == labels).sum().item()
            count += batch_size
            predictions.extend(batch_predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "predictions": predictions,
        "targets": targets,
    }

