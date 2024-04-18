import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split


label_map = {
    0: "Road",
    2: "Sidewalk",
    4: "Person",
    5: "Rider",
    6: "Motorcycle",
    7: "Bicycle",
    9: "Car",
    10: "Truck",
    11: "Bus",
    12: "Train",
    14: "Wall",
    15: "Fence",
    18: "Traffic Sign",
    19: "Traffic Light",
    20: "Pole",
    22: "Building",
    24: "Vegetation",
    25: "Sky",
}


colour_map = {
    0: (128, 64, 128),
    2: (244, 35, 232),
    4: (220, 20, 60),
    5: (255, 0, 0),
    6: (0, 0, 230),
    7: (119, 11, 32),
    9: (0, 0, 142),
    10: (0, 0, 70),
    11: (0, 60, 100),
    12: (0, 80, 100),
    14: (102, 102, 156),
    15: (190, 153, 153),
    18: (220, 220, 0),
    19: (250, 170, 30),
    20: (153, 153, 153),
    22: (70, 70, 70),
    24: (107, 142, 35),
    25: (70, 130, 180),
}


class Custom_IDD_dataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_dir="image_archive",
        mask_dir="mask_archive",
        transform=None,
        target_transform=None,
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.num_samples = len(os.listdir(self.image_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = f"image_{idx}.jpg"
        mask_name = f"mask_{idx}.jpg"

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def gen_freqs(dataset):
    itrs = range(len(dataset))
    freqs = np.zeros(256)
    for i in tqdm(itrs, desc="generating label counts:"):
        _, mask = dataset[i]
        m_ray = np.asarray(mask).flatten()
        bin_counts = np.bincount(m_ray, minlength=256)
        freqs += bin_counts
    return freqs


def gen_colour_masks(dataset):
    dir = os.path.join(dataset.root_dir, "colour_masks")
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)

    itrs = range(len(dataset))
    for i in tqdm(itrs, desc="generating coloured masks:"):
        _, mask = dataset[i]
        m_ray = np.asarray(mask)
        c_mask = np.zeros((m_ray.shape[0], m_ray.shape[1], 3), dtype=np.uint8)
        for label_value, color in colour_map.items():
            c_mask[m_ray == label_value] = color

        c_mask_img = Image.fromarray(c_mask)
        c_mask_img.save(os.path.join(dir, f"mask_{i}.jpg"))
    print(f"coloured masks saved at {dir}")
    return


def place_img(data_dir, dataset):

    image_dir = os.path.join(data_dir, "image_archive")
    mask_dir = os.path.join(data_dir, "masks_archive")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    else:
        shutil.rmtree(image_dir)
        os.makedirs(image_dir)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    else:
        shutil.rmtree(mask_dir)
        os.makedirs(mask_dir)

    for i in tqdm(range(len(dataset))):
        img, mask = dataset[i]

        image_path = os.path.join(image_dir, f"image_{i}.jpg")
        mask_path = os.path.join(mask_dir, f"mask_{i}.jpg")

        img.save(image_path)
        mask.save(mask_path)


tf_img = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

tf_mask = transforms.Compose(
    [
        transforms.Resize((512, 512)),
    ]
)


def setup(gen_mask=False, get_freq=False, test_data=False):
    dataset = Custom_IDD_dataset("IDD20K_II")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    freqs = None
    if gen_mask:
        gen_colour_masks(dataset)
    if get_freq:
        freqs = gen_freqs(dataset)
    dataset_col = Custom_IDD_dataset("IDD20K_II", mask_dir="colour_masks")
    dataloader_col = DataLoader(dataset_col, batch_size=32, shuffle=True)

    if test_data:
        train_len = int(0.7 * len(dataset))
        test_len = int(len(dataset) - train_len)
        _, test_data = random_split(dataset, (train_len, test_len))
        place_img("test_data", test_data)

    test_data = Custom_IDD_dataset(
        "test_data",
        mask_dir="masks_archive",
        transform=tf_img,
        target_transform=tf_mask,
    )
    return dataset_col, test_data, freqs
