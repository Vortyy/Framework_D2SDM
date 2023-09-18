import os
import torch

import torchvision.transforms as T

from torch.utils.data import Dataset, Subset
from numpy import add
from torchvision.datasets import CIFAR100, CIFAR10, MNIST

Datasets = {
    "MNIST" : MNIST,
    "CIFAR10" : CIFAR10,
    "CIFAR100" : CIFAR100
}

Transformations = {
    "toTensor" : T.ToTensor(),
    "classic" : T.Compose([T.ToTensor(), T.Lambda(lambda x : torch.flatten(x))]),
    "pretrained" : T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
}

def generate_task_list(nb_task : int, step : int) -> list[list] :
    return [[i for i in range(n, n+step)] for n in range(0, nb_task * step, step)]

def get_dataset_pytorch(name : str, transformation : str) -> list[Dataset, Dataset] :
    train_data = Datasets[name](os.getcwd() + "/data/" + name, train=True, transform=Transformations[transformation], download=True)
    test_data = Datasets[name](os.getcwd() + "/data/" + name, train=False, transform=Transformations[transformation], download=True)
    return [train_data, test_data]

def get_subset_class(data : Dataset, id_label_start : int, step : int, limit : int = None) -> Subset :
    idx_class = 0 * len(data)

    if(type(data.targets) is not torch.Tensor) :
        targets = torch.tensor(data.targets)
    else :
        targets = data.targets
            
    for id_class in range(id_label_start, id_label_start + step) :
        idx_temp = (targets == id_class)
        idx_class = add(idx_class, idx_temp)

    idx_class = idx_class.nonzero().reshape(-1)

    if(limit != None) :
        idx_class = idx_class[:limit]

    data_sub = Subset(data, idx_class)

    return data_sub

def get_subset_class_list(data : Dataset, id_label_list : list[int], limit : int = None) -> Subset :
    idx_class = 0 * len(data)

    if(type(data.targets) is not torch.Tensor) :
        targets = torch.tensor(data.targets)
    else :
        targets = data.targets
            
    for id_class in id_label_list :
        idx_temp = (targets == id_class)
        idx_class = add(idx_class, idx_temp)

    idx_class = idx_class.nonzero().reshape(-1)

    if(limit != None) :
        idx_class = idx_class[:limit]

    data_sub = Subset(data, idx_class)

    return data_sub