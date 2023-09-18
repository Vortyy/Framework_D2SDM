import torch
from model.d2sdm import d2sdm
from types import SimpleNamespace
import torch.nn.functional as F
import wandb

class D2sdm_smooth(d2sdm) :
    def __init__(self, params : SimpleNamespace, device : str, logger : wandb = None) -> None:
        super(D2sdm_smooth, self).__init__(params, device, logger)
        self.rt = params.rt
        
    def training_step(self, X : torch.Tensor, y : torch.Tensor) -> None:
        if(self.encoder != None) :
            X = self.encoder_transform(X)
            X = self.encoder(X)

        #Work on data that make distance calculation easier
        X = torch.reshape(X, (1, X.size(0), X.size(1)))
        y = F.one_hot(y, num_classes=self.params.num_classes).type(torch.float)

        if(self.A != None) :
            dist = self.distance_fct(X, self.A.data, **self.params.distance_params)#Using data make program avoid to account this action for the gradient
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
                self.A.data = self.A.data + self.params.lr * torch.mul(self.A.data - X[0, idx.item()], torch.reshape(idx_soft, (idx_soft.size(0), 1, 1)))
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