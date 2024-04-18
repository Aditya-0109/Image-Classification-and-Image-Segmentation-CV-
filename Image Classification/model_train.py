import torch.nn as nn
import torch
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

train_loader, val_loader = get_data(0, loader=True), get_data(1, loader=True)


class ConvNet(BaseClassifier):
    def __init__(self, classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        dim_flat = config["resize"] // 16
        self.fc = nn.Linear(128 * dim_flat * dim_flat, classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def setup():
    train_loader, val_loader = (
        get_data(value=0, loader=True),
        get_data(value=2, loader=True),
    )
    model = ConvNet().to(device)
    model = model_init_train(model, config, train_loader, val_loader, "ConvNet.onnx")


if __name__ == "__main__":
    setup()
