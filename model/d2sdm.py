import torch
from torch import nn
from types import SimpleNamespace
import torch.nn.functional as F
from utils.distances import distances, metric_tvMF
from sklearn.neighbors import LocalOutlierFactor
from utils.encoder import encoder_fct
import wandb

class d2sdm(nn.Module) :
    def __init__(self, params : SimpleNamespace, device : str, logger : wandb = None) -> None:
        super(d2sdm, self).__init__()

        self.distance_fct = distances[params.distance_fct]
        self.nb_pruned = int(params.nb_A_max - params.p_prunning * params.nb_A_max)

        self.A = None #A = torch.tensor((n_neurons, 1, n_features)) #Adresse Tensors -> we gave a real image of 1 just to start The middle dim allow to calculate Cosine Sime over a batch 
        self.C = None #C = torch.tensor((n_neurons, n_classes)) #Content Tensors ici les content on id :)

        self.params = params
        self.device = device
        self.logger = logger

        self.encoder, self.encoder_transform = encoder_fct[params.encoder_name](device=device) if(params.encoder_name != None) else (None, None)

    def forward(self, X : torch.Tensor, already_encoded : bool = False) -> torch.Tensor :
        if(self.encoder != None and not already_encoded) :
            X = self.encoder_transform(X)
            X = self.encoder(X)

        X = torch.reshape(X, (1, X.size(0), X.size(1))) #to make calculation possible between neurons and inputs.
        pred = self.distance_fct(X, self.A, **self.params.distance_params)

        #Top-k essay : --------------------------------------------------------------------
        # vals, _ = torch.topk(pred, k=20, dim=0)
        # inhib_amount, _ = torch.min(vals.detach(), dim=0)
        # pred = F.relu(X - inhib_amount)
        #-------------------- Comment here if need to remove top-K ------------------------

        pred = F.softmin(pred / self.params.beta, dim=0)
        pred = F.linear(pred.T, self.C.T, bias=None)
        return pred
    
    def prune(self) -> None :
        if(self.params.prunning_mod != None) :
            classifier = LocalOutlierFactor(n_neighbors=self.params.n_neighbors, metric="cosine", n_jobs=-1)
            classifier.fit_predict(self.A.data.reshape(self.A.data.size(0), self.A.data.size(2)).cpu())
            LofNegScore = classifier.negative_outlier_factor_
            #print(classifier.effective_metric_)
            TensorScore = torch.tensor(LofNegScore).to(self.device)
            idx = None

            if(self.params.prunning_mod == "naive") :
                _ , idx = torch.topk(TensorScore, k=self.nb_pruned)

            elif(self.params.prunning_mod == "mean") :
                idx = torch.tensor([]).to(self.device)
                tensor_argmax = torch.argmax(self.C, dim=1).to(self.device)

                idx_value_i = []
                for i in range(self.C.size(1)) :
                    idx_ = (tensor_argmax == i).nonzero().flatten()
                    if(idx_.numel() != 0) :
                        idx_value_i.append(idx_)
                
                k = int(self.nb_pruned / len(idx_value_i))

                for i in idx_value_i : 
                    if(len(i) > k) :
                        _, best_idx = torch.topk(TensorScore[i], k=k)
                        idx = torch.cat((idx, i[best_idx]), dim=0)
                    else :
                        idx = torch.cat((idx, i), dim=0)

                self.A.data = torch.index_select(self.A.data, 0, idx.to(torch.int32))
                self.C.data = torch.index_select(self.C.data, 0, idx.to(torch.int32))

            elif(self.params.prunning_mod == "new") :
                pass

    def training_step() :
        raise NotImplementedError()

    def grad_switch(self) -> None :
        self.A.requires_grad = not self.A.requires_grad
        self.C.requires_grad = not self.C.requires_grad

    def log(self, dict : dict) -> None :
        if(self.logger != None) :
            self.logger.log(dict)