import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import Accuracy

def get_accuracy(model, loader : DataLoader, task_id : int, device : str, name : str = None, not_displayed : bool = False) -> torch.Tensor :
    """
    this function compute the accuracy of a given model, onto a given Dataset.
    the optional args : 'name', 'not_displayed' are just here to make thing more clear

    return : the accuracy of the model along the dataset in a Tensor
    """
    task_pred = []
    task_target = []

    for (X, y) in tqdm(loader, desc=f"{name} on task {task_id} ", disable=not_displayed) :
        task_target.append(y)
        task_pred.append(model(X.to(device)))

    tensor_targets = torch.cat(task_target).to(device)
    tensor_preds = torch.cat(task_pred).to(device)

    metric = Accuracy(task="multiclass", num_classes=model.params.num_classes).to(device)
    accuracy = metric(tensor_preds, tensor_targets)

    return accuracy