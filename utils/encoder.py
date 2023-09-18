import torch

from torchvision.models import ResNet18_Weights, resnet18, resnet50, ResNet50_Weights, ResNet, resnet34, ResNet34_Weights
from torchvision.transforms._presets import ImageClassification

def init_resnet18(device : str) -> tuple[ResNet, ImageClassification] :
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device=device)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, transform

def init_resnet34(device : str) -> tuple[ResNet, ImageClassification] :
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device=device)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()

    transform = ResNet34_Weights.IMAGENET1K_V1.transforms()
    return model, transform

def init_resnet50(device : str) -> tuple[ResNet, ImageClassification]:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device=device)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()

    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    return model, transform

encoder_fct = {
    "resnet18" : init_resnet18,
    "resnet34" : init_resnet34,
    "resnet50" : init_resnet50
}