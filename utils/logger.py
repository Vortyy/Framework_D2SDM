import wandb
import torch
import umap
import warnings
import umap.plot
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

def init_wandb(p_key : str, name : str, config : dict, offline : bool=False) -> wandb :
    '''
    Function that init a wandb object that will be used during the run by the model and the handler
    '''

    wandb.login(key=p_key)

    return wandb.init(
                project="D2SDM",
                name=name,
                config=config
            )   

def end_log(logger : wandb, stat_recorder : dict, epochs_per_task : int) -> None :
    test = [[x, y] for (x, y) in zip(stat_recorder["val_last_acc"], [i / (epochs_per_task) for i in range(len(stat_recorder["val_last_acc"]))])]
    t = wandb.Table(columns=["val_last_acc", "task_learning"], data=test)

    logger.log({"Validation/val_last_acc" : wandb.plot.line(table=t, x="task_learning", y="val_last_acc", title="validation last acc along task learning")})

                # tensor_test = torch.tensor(stat_recorder["ind_task_acc"])
                # arr = np.array(tensor_test)

                # logger.log({"Additional/stat_ind_taks_acc" : wandb.plot.line_series(
                #     xs = [i / (opt.epochs_per_task + 1) for i in range(len(stat_recorder["ind_task_acc"]))],
                #     ys = arr.transpose(),
                #     keys= ["val_acc_t1","val_acc_t2", "val_acc_t3", "val_acc_t4", "val_acc_t5"],
                #     xname= "task_learned",
                #     title= "Validation accuracy of each task during the learning"
                # )})

    tensor_la = torch.tensor(stat_recorder["last_acc"], device="cpu")
    last_acc_array = np.array(tensor_la)

    logger.summary["Test/avg_acc"] = np.sum(last_acc_array, axis=0)[1] / len(last_acc_array)
    logger.summary["Test/last_acc"] = last_acc_array[len(last_acc_array) - 1][1]

    table = wandb.Table(columns=["task", "acc"], data=last_acc_array)
    logger.log({"Test/last_acc_plot" : wandb.plot.line(table=table, x="task", y="acc", title="plot accuracy along task learning")})
                
    time.sleep(60) #Adding time to wait that upload everything to avoid a finish to soon without synching... WANDB problems

def umap_plot(model, save_name : str, n_components : int = 2, n_neighbors : int = 20, metric : str or function = "euclidean", min_dist : float = 0.01) -> None :
    umap_tool = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist
    )

    visualize_data = model.A.data
    visualize_data = torch.reshape(visualize_data, (visualize_data.size(0), visualize_data.size(2)))

    labels = torch.argmax(model.C.data, dim=1)

    labels = labels.to("cpu")
    visualize_data = visualize_data.to("cpu")

    mapper = umap_tool.fit(visualize_data)
    umap.plot.points(mapper, labels=labels)
    plt.savefig(save_name)

def dir_verif(path : str) -> None:
    if(not os.path.exists(path)) :
        os.mkdir(path) 
    

def warning_ignore() -> None :
    '''
    Set all the warning filter off to get more visibility on the ongoing of the run.
    '''
    warnings.filterwarnings("ignore")
    warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)
    warnings.filterwarnings('ignore', category=NumbaDeprecationWarning, append=True)