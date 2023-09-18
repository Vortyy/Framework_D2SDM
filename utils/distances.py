import torch

import torch.nn.functional as F
import numpy as np

from numba import njit

def _normalize(X: torch.Tensor) -> torch.Tensor :
    '''
    center and normalize a tensor based on it's first dimension of features code from Alex/Rowan codes.
    '''
    return (X - torch.mean(X, dim=1).view(-1, 1)) / torch.std(X, dim=1).view(-1, 1)

def compute_corr_distance(X: torch.Tensor, Y: torch.Tensor, **args) -> torch.Tensor :
    '''
    require args : None

    distances computed as used in the code of Alex/Rowan that compute a correlated distance between 2 tensors

    return : the squared exp of this distance
    '''
    D = X.shape[2]
    X = _normalize(X.reshape((X.size(1), X.size(2))))
    Y = _normalize(Y.reshape((Y.size(0), Y.size(2))))

    Z = 1 - (torch.mm(X, Y.T) / (D - 1)).abs()
    return torch.exp(Z).pow(2).T

def distance_cosine(A : torch.Tensor, B : torch.Tensor, **args) -> torch.Tensor :
    '''
    required args : 'dim'

    distance_cosine return the distance cosine between 2 tensors, using the dimension selected to calculate it.
    Using torch.nn.functionnal.cosine_similarity() : https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html?highlight=cosine#torch.nn.functional.cosine_similarity \\
    it compute 1 - cosine_similarity() -> d_cos [0, 2] where 0 is similar and 2 is the exact opposite.

    return : distance_cosine in a tensor
    '''
    return 1 - F.cosine_similarity(A, B, dim=args["dim"])

def norm_p(A : torch.Tensor, B : torch.Tensor, **args) -> torch.Tensor :
    '''
    required args : 'p' 
    
    distance-p calculate between 2 tensors using torch.cdist : https://pytorch.org/docs/stable/generated/torch.cdist.html

    return : distance L_n in a Tensor
    '''
    return torch.cdist(A, B, p=args["p"]).flatten(start_dim=1)

def t_vMF(A : torch.Tensor, B : torch.Tensor, **args) -> torch.Tensor :
    ''' 
    required args : 'dim', 'k'

    distances suivant la distribution student-t de von-Mises Fischer lien du papier : https://openaccess.thecvf.com/content/CVPR2021/html/Kobayashi_T-vMF_Similarity_for_Regularizing_Intra-Class_Feature_Distribution_CVPR_2021_paper.html \\
    it takes 2 tensors and compute the t-vMF similarity between them.

    return : t-vMF similarity in a Tensor
    '''
    cos = F.cosine_similarity(A, B, dim=args["dim"])
    return 2 - ((1 + cos)/(1 + args["k"] * (1 - cos)))

distances = {
    "cos" : distance_cosine,
    "norm_p" : norm_p,
    "t-vMF" : t_vMF,
    "corr" : compute_corr_distance
}

@njit
def metric_tvMF(A : np.ndarray, B : np.ndarray, k : float = 8.) -> np.ndarray :
    '''
    require args : k -> that are the concentration constant used to compute the distances.

    This function allow us to use the t-vMF similarity in the metric parameter of LoF class from sklearn because it precompile it. 
    The main purpose of this funciton is given 2-ndarray return their t-vMF similarity.
    
    return : t-vMF similarity in an ndarray
    '''
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    cos = np.dot(A, B)
    return 2 - ((1 + cos)/(1 + k * (1 - cos)))

if __name__ == "__main__" :
    A = torch.tensor([[2., 1.], [0. , 1.]])
    B = torch.tensor([[1., 1.]])

    result = distance_cosine(A, B)
    print(result) # 1 - [0.9487, 0.7071] = [0.0513, 0.2929]
    pass