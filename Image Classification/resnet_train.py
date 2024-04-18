import torch.nn as nn
import torch
from torchvision import models
from dataset_class import get_data, config
from dataset_train import BaseClassifier, model_init_train

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available() else "cpu"
)


class ResNet18(BaseClassifier, nn.Module):
    def __init__(self, classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in self.resnet18.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, classes)

    def forward(self, x):
        return self.resnet18(x)


def setup(hyp, train_data, val_data, model_out):
    model = ResNet18().to(device)
    model = model_init_train(model, hyp, train_data, val_data, model_out)


if __name__ == "__main__":
    train_loader, val_loader = get_data(0, loader=True), get_data(1, loader=True)
    setup(config, train_loader, val_loader, "ResNet.onnx")
