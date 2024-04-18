from dataset_class import get_data, config
from torch.utils.data import DataLoader, ConcatDataset
from resnet_train import setup
import torchvision.transforms as transforms

aug_hf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((config["resize"], config["resize"])),
        transforms.ToTensor(),
    ]
)

aug_rt = transforms.Compose(
    [
        transforms.RandomRotation(degrees=15),
        transforms.Resize((config["resize"], config["resize"])),
        transforms.ToTensor(),
    ]
)

aug_rc = transforms.Compose(
    [
        transforms.Resize((config["resize"], config["resize"])),
        transforms.RandomCrop(size=(300, 300), padding=4),
        transforms.Resize((config["resize"], config["resize"])),
        transforms.ToTensor(),
    ]
)


def gen_augments(value):
    og_data = get_data(value)
    hf_data = get_data(value, tf=aug_hf)
    rt_data = get_data(value, tf=aug_rt)
    rc_data = get_data(value, tf=aug_rc)

    fin_data = ConcatDataset([og_data, hf_data, rt_data, rc_data])
    fin_loader = DataLoader(fin_data, batch_size=config["batch_size"], shuffle=True)
    return fin_loader


if __name__ == "__main__":
    train_loader = gen_augments(0)
    val_loader = gen_augments(1)

    setup(config, train_loader, val_loader, "ResNet_aug.onnx")
