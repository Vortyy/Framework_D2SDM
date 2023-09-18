import os

from types import SimpleNamespace
from datetime import datetime

from model.d2sdm_smooth import D2sdm_smooth
from model.hdsdm import Hdsdm

from run.trainer_classic import trainer_classic

from utils.logger import init_wandb, warning_ignore, dir_verif
from utils.datasets import get_dataset_pytorch

# Global variables :
EXP_NAME = "run_DSDM_1t_"
LOGGER = False
SAVE_MODEL = False
DEVICE = "cuda:0" if(torch.cuda.is_available()) else "cpu"
VERBOSE = 0 #Not handled yet.
NB_RUNS = 1

#Configuration vars model and handler :
config = {
    "data" : {
        "ds" : "CIFAR10",
        "transformation" : "toTensor"
    },
    "handler_opt" : {
        "nb_task" : 3,
        "step" : 2,
        "nb_element" : None,
        # "splitting_criterion" : 1., # x * forward + (1 - x) * backward
        "bs" : 32,
        # "lr" : 1,
        "epochs_per_task" : 1
    },
    "model_opt" : {
        "distance_fct": "cos",
        "distance_params" : {"dim" : 2},
        "nb_A_max": 5000,
        "num_classes" : 10,
        "encoder_name" : "resnet18",
        #"k" : 40,
        "p_prunning": 0.03,
        "prunning_mod": "mean",
        # "high_boundary" : 0.5,
        # "low_boundary" : 0.2,
        "rt" : 0.2,
        "lr": 0.001,
        # "loss_fn_name" : "CE",
        "lr_rt": 0.001,
        "beta": 0.01,
        "beta_bcT" : 1,
        "n_neighbors" : 20 #4000 // 10 -> 400 - 50 au voisinage ici on tombe a 350 
    }
}

if __name__ == "__main__" :    

    warning_ignore()

    for i in range(NB_RUNS) :
        name = EXP_NAME + datetime.now().strftime("%d%m%Y_%H%M%S")

        train_data, test_data = get_dataset_pytorch(config["data"]["ds"], config["data"]["transformation"])

        if(LOGGER) :
            p_key = os.environ.get("WANDB_API_KEY")
            logger = init_wandb(p_key=p_key, name=name, config=config)
        else :
            logger = None

        model = Hdsdm(params=SimpleNamespace(**config["model_opt"]), device=DEVICE, logger=logger).to(DEVICE)

        handler = trainer_classic(opt=SimpleNamespace(**config["handler_opt"]), device=DEVICE, logger=logger)
        handler.fit_and_test(model, train_data, test_data)

        if(SAVE_MODEL) :
            path = "./data/model_saved/" + name + ".pt"
            torch.save(model, path)
            print(f"model succesfully saved at : {path}")

        if(LOGGER) :
            logger.finish()