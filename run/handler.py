import torch
import wandb
import os

from datetime import datetime
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn
from types import SimpleNamespace
from utils.datasets import get_subset_class_list, generate_task_list
from utils.metric import get_accuracy
from utils.logger import end_log, umap_plot, dir_verif

class Handler() :
    def __init__(self, opt : SimpleNamespace, device : str, logger : wandb = None) -> None:
        self.opt = opt
        self.logger = logger
        self.device = device

        self.stat_recorder = {
            "last_acc" : [],
            "val_last_acc" : [],
            "ind_task_acc" : []
        }

        self.timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    def fit_and_test(self, model, train_data : Dataset, test_data : Dataset) -> None :
        # sub_add_list = []
        labels_list = generate_task_list(self.opt.nb_task, self.opt.step)
        labels_test = []

        print(labels_list)

        for task_id, labels in enumerate(labels_list) :
            labels_test += labels

            subset_train_data = get_subset_class_list(train_data, labels, limit=self.opt.nb_element) #Attention le parametre de limit a ete enlever :)
            subset_test_data = get_subset_class_list(test_data, labels_test) #Attention le parametre de limit a ete enlever :)

            # subset_train_data, subset_val_data = random_split(subset_train_data, [int(len(subset_train_data) * 0.9), len(subset_train_data) - int(len(subset_train_data) * 0.9)])
            subset_test_data, subset_val_data = random_split(subset_test_data, [int(len(subset_test_data) * 0.9), len(subset_test_data) - int(len(subset_test_data) * 0.9)])

            # sub_add_list.append(subset_add_data)

            self.train(model, subset_train_data, subset_val_data, None, task_id)
            self.end_training_task_hook(model, task_id)

            self.test(model, subset_test_data, task_id)
            self.end_testing_task_hook(model, task_id)

    def train() -> None :
        raise NotImplementedError

    def validation(self, model, val_data : Dataset, task_id : int) -> torch.Tensor :
        val_loader = DataLoader(val_data, batch_size=self.opt.bs)

        val_last_acc = get_accuracy(model, val_loader, task_id, self.device, "Validation")

        print(val_last_acc)
        self.stat_recorder["val_last_acc"].append(val_last_acc)
        return val_last_acc
            
    def test(self, model, test_data : Dataset, task_id : int) -> None :
        test_loader = DataLoader(test_data, batch_size=self.opt.bs)

        accuracy = get_accuracy(model, test_loader, task_id, self.device, "Test")

        print(accuracy)
        self.stat_recorder["last_acc"].append([task_id, accuracy])

        if(self.opt.nb_task == (task_id + 1) and self.logger != None) :
            end_log(self.logger, self.stat_recorder, self.opt.epochs_per_task)

    # ----------------------------------------- Hook ----------------------------------------------#

    def end_training_task_hook(self, model, task_id : int) -> None :
        dir_verif(f"data/results/{self.timestamp}")

        umap_plot(model, f"data/results/{self.timestamp}/map_task_{task_id}")

    def end_testing_task_hook(self, model, task_id : int) -> None :
        pass
