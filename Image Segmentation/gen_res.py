from torch import nn
from DeepLabV3Plus import network
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from dataset_class import Custom_IDD_dataset, colour_map
from DeepLabV3Plus.datasets import Cityscapes

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps:0" if torch.backends.mps.is_available() else "cpu"
)


def get_model(device):
    model = network.modeling.__dict__["deeplabv3plus_mobilenet"](
        num_classes=19, output_stride=16
    )
    checkpoint = torch.load(
        "best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["model_state"])

    model = nn.DataParallel(model)
    model.to(device)
    return model


def colour_to_label(colour_mask):
    label_mask = np.full(
        (colour_mask.shape[0], colour_mask.shape[1]), 255, dtype=np.uint8
    )
    for label_value, color in colour_map.items():
        mask = np.all(colour_mask == color, axis=-1)
        label_mask[mask] = label_value
    return label_mask


def gen_colour_masks(label_mask):
    m_ray = np.asarray(label_mask)
    c_mask = np.zeros((m_ray.shape[0], m_ray.shape[1], 3), dtype=np.uint8)
    for label_value, color in colour_map.items():
        c_mask[m_ray == label_value] = color
    c_mask_img = Image.fromarray(c_mask)
    return c_mask_img


def run_inf(model, test_data, dir):
    res_fldr = dir + "/res_data"
    os.makedirs(res_fldr)
    with torch.no_grad():
        model = model.eval()

        for i in tqdm(range(len(test_data))):
            img, _ = test_data[i]
            img = img.unsqueeze(0)
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]
            pred = Cityscapes.decode_target(pred).astype("uint8")
            label_mask = colour_to_label(pred)
            label_preds = Image.fromarray(label_mask)
            label_preds.save(os.path.join(res_fldr, f"mask_{i}.jpg"))

    res_data = Custom_IDD_dataset(dir, image_dir="image_archive", mask_dir="res_data")
    return res_data


def get_res(test_data, dir, get_data=False):
    if get_data:
        res_data = Custom_IDD_dataset(
            dir, image_dir="image_archive", mask_dir="res_data"
        )
        return res_data
    model = get_model(device)
    return run_inf(model, test_data, dir)
