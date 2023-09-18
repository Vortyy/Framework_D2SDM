import torch
from torch import nn
from types import SimpleNamespace
import torch.nn.functional as F
from utils.distances import distances, metric_tvMF
from sklearn.neighbors import LocalOutlierFactor
from utils.encoder import encoder_fct
from geoopt.manifolds import Stereographic
import wandb

class Hdsdm(nn.Module) :
    def __init__(self, params : SimpleNamespace, device : str, logger : wandb = None) -> None:
        super(Hdsdm, self).__init__()

        self.nb_pruned = int(params.nb_A_max - params.p_prunning * params.nb_A_max)

        self.A = None #A = torch.tensor((n_neurons, 1, n_features)) #Adresse Tensors -> we gave a real image of 1 just to start The middle dim allow to calculate Cosine Sime over a batch 
        self.C = None #C = torch.tensor((n_neurons, n_classes)) #Content Tensors ici les content on id :)

        self.params = params
        self.device = device
        self.logger = logger

        self.encoder, self.encoder_transform = encoder_fct[params.encoder_name](device=device) if(params.encoder_name != None) else (None, None)
        self.manifold = Stereographic(k=-1)
        self.rt = self.params.rt

    def forward(self, X : torch.Tensor, already_encoded : bool = False) -> torch.Tensor :
        if(self.encoder != None and not already_encoded) :
            X = self.encoder_transform(X)
            X = self.encoder(X)

        X = self.manifold.projx(X, dim=1)
        
        X = torch.reshape(X, (1, X.size(0), X.size(1)))

        pred = self.manifold.dist(self.A, X)

        pred = F.softmin(pred / self.params.beta, dim=0)
        pred = F.linear(pred.T, self.C.T, bias=None)
    
        return pred
    
    def prune(self) -> None :
        if(self.params.prunning_mod != None) :
            classifier = LocalOutlierFactor(n_neighbors=self.params.n_neighbors, metric="minkowski", n_jobs=-1)

            O = self.manifold.origin((1, 1, 512)).to(self.device)
            log_A = self.manifold.logmap(self.A.data, O)
            log_A = torch.reshape(log_A, (log_A.size(0), log_A.size(2)))
            
            classifier.fit_predict(log_A.cpu())
            LofNegScore = classifier.negative_outlier_factor_
            #print(classifier.effective_metric_)
            TensorScore = torch.tensor(LofNegScore).to(self.device)
            idx = None

            if(self.params.prunning_mod == "naive") :
                _ , idx = torch.topk(TensorScore, k=self.nb_pruned, largest=False)

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
                        _, best_idx = torch.topk(TensorScore[i], k=k, largest=False)
                        idx = torch.cat((idx, i[best_idx]), dim=0)
                    else :
                        idx = torch.cat((idx, i), dim=0)

                self.A.data = torch.index_select(self.A.data, 0, idx.to(torch.int32))
                self.C.data = torch.index_select(self.C.data, 0, idx.to(torch.int32))

            elif(self.params.prunning_mod == "new") :
                pass

    def training_step(self, X : torch.Tensor, y : torch.Tensor) -> None:
        if(self.encoder != None) :
            X = self.encoder_transform(X)
            X = self.encoder(X)

        X = self.manifold.projx(X, dim=1)
        
        X = torch.reshape(X, (1, X.size(0), X.size(1)))
        y = F.one_hot(y, num_classes=self.params.num_classes).type(torch.float)

        if(self.A != None) :
            dist = self.manifold.dist(self.A, X)
            dist_soft = F.softmin(dist / self.params.beta, dim=0)

            dBmu, _ = torch.min(dist, dim=0)
            mean = torch.mean(dBmu) #torch.topk(dBmu, k=5, largest=False)[0])

            rt_check = (dBmu > self.rt)

            idx_sup = rt_check.nonzero().flatten()
            idx_inf = rt_check.logical_not().nonzero().flatten() 

            self.sharpening(X, y, dist_soft, idx_inf)
            self.concatenate(X, y, idx_sup)

            self.rt = self.params.lr_rt * self.rt + (1.-self.params.lr_rt) * mean.item()

            self.log({
                "model/rt" : self.rt,
                "model/nb_neurons" : self.A.data.size(0)
            })
            
        else : 
            self.A = torch.nn.Parameter(torch.reshape(X, (X.size(1), 1, X.size(2))), requires_grad=False).to(self.device)
            self.C = torch.nn.Parameter(y, requires_grad=False).to(self.device)

    def sharpening(self, X : torch.Tensor, y : torch.Tensor, dist_soft : torch.Tensor, idx_inf : torch.Tensor) -> None :
        #iterative way : 
        if(idx_inf.nelement() != 0) :
            for idx in idx_inf :
                idx_soft = dist_soft[:, idx.item()]
                A_diff = self.manifold.mobius_sub(self.A.data, X[0, idx.item()])
                wei_A_diff = self.manifold.mobius_scalar_mul(torch.reshape(idx_soft, (idx_soft.size(0), 1, 1)), A_diff)
                lr_A_diff = self.manifold.mobius_scalar_mul(torch.tensor(self.params.lr), wei_A_diff)
                self.A.data += self.manifold.mobius_add(self.A.data, lr_A_diff)

                self.C.data = self.C.data + self.params.lr * torch.mul(self.C.data - y[idx.item()], torch.reshape(idx_soft, (idx_soft.size(0), 1)))

    def concatenate(self, X : torch.Tensor, y : torch.Tensor, idx_sup : torch.Tensor) -> None :
        if(idx_sup.nelement() != 0) :
            if(self.A.data.size(0) < self.params.nb_A_max) :
                X_sup = torch.index_select(X, dim=1, index=idx_sup)
                y_sup = torch.index_select(y, dim=0, index=idx_sup)

                self.A.data = torch.cat((self.A.data, X_sup.reshape(X_sup.size(1), 1, X_sup.size(2))), dim=0) #Reshape input for adaptating to addresse dimension.
                self.C.data = torch.cat((self.C.data, y_sup), dim=0)
                
            else :
                self.prune()

    def grad_switch(self) -> None :
        self.A.requires_grad = not self.A.requires_grad
        self.C.requires_grad = not self.C.requires_grad

    def log(self, dict : dict) -> None :
        if(self.logger != None) :
            self.logger.log(dict)