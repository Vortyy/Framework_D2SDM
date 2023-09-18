from run.handler import Handler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from types import SimpleNamespace
import wandb

class trainer_classic(Handler) :
    def __init__(self, opt: SimpleNamespace, device: str, logger: wandb = None) -> None:
        super().__init__(opt, device, logger)

    def train(self, model, train_data : Dataset, val_data : Dataset, add_data : list[Dataset] or None, task_id : int):
        train_loader = DataLoader(train_data, batch_size=self.opt.bs)

        for _ in range(self.opt.epochs_per_task) :
            for (X, y) in tqdm(train_loader, desc=f"training on task {task_id} ") :
                model.training_step(X.to(self.device), y.to(self.device))
            
            self.validation(model, val_data, task_id)
        
