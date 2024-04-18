import os
import shutil
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

labels_RW = {
    "amur_leopard": 0,
    "amur_tiger": 1,
    "birds": 2,
    "black_bear": 3,
    "brown_bear": 4,
    "dog": 5,
    "roe_deer": 6,
    "sika_deer": 7,
    "wild_boar": 8,
    "people": 9,
}


classes_RW = [
    "amur_leopard",
    "amur_tiger",
    "birds",
    "black_bear",
    "brown_bear",
    "dog",
    "roe_deer",
    "sika_deer",
    "wild_boar",
    "people",
]

config = dict(
    epochs=10,
    classes=10,
    kernels=[3, 3, 3],
    batch_size=32,
    learning_rate=0.001,
    dataset="RUSSIAN_WILDLIFE",
    architecture="CNN",
    resize=512,
)

tf = transforms.Compose(
    [
        transforms.Resize((config["resize"], config["resize"])),
        transforms.ToTensor(),
    ]
)

src = "Cropped_final"

dst_tr = "data_trn"
dst_vl = "data_val"
dst_ts = "data_tst"


def gen_dataset(src, dst_tr, dst_ts, dst_vl):
    if not os.path.exists(dst_tr):
        os.makedirs(dst_tr)
    else:
        shutil.rmtree(dst_tr)
        os.makedirs(dst_tr)

    if not os.path.exists(dst_ts):
        os.makedirs(dst_ts)
    else:
        shutil.rmtree(dst_ts)
        os.makedirs(dst_ts)

    if not os.path.exists(dst_vl):
        os.makedirs(dst_vl)
    else:
        shutil.rmtree(dst_vl)
        os.makedirs(dst_vl)

    ann_lst = []
    for file, labls in labels_RW.items():
        ani_dir = src + "/" + file
        for img in os.listdir(ani_dir):
            new_entry = (img, labls)
            ann_lst.append(new_entry)

    ann_df = pd.DataFrame(ann_lst)
    train_data, temp_data = train_test_split(
        ann_df, test_size=0.3, random_state=42, stratify=ann_df[1]
    )
    test_data, val_data = train_test_split(
        temp_data, test_size=0.33, random_state=42, stratify=temp_data[1]
    )

    for _, row in train_data.iterrows():
        src_file = os.path.join(src, classes_RW[row[1]], str(row[0]))
        shutil.copy(src_file, dst_tr)
    train_data.to_csv("anno_train.csv", index=False, header=False)

    for _, row in test_data.iterrows():
        src_file = os.path.join(src, classes_RW[row[1]], str(row[0]))
        shutil.copy(src_file, dst_ts)
    test_data.to_csv("anno_test.csv", index=False, header=False)

    for _, row in val_data.iterrows():
        src_file = os.path.join(src, classes_RW[row[1]], str(row[0]))
        shutil.copy(src_file, dst_vl)
    val_data.to_csv("anno_val.csv", index=False, header=False)


class Custom_RW_dataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file, names=[0, 1])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_data(value=0, tf=tf, loader=False):
    if value == 0:
        data = Custom_RW_dataset("anno_train.csv", "data_trn", tf)
    elif value == 1:
        data = Custom_RW_dataset("anno_test.csv", "data_tst", tf)
    else:
        data = Custom_RW_dataset("anno_val.csv", "data_val", tf)

    if loader:
        data_loaded = DataLoader(data, batch_size=config["batch_size"], shuffle=True)
        return data_loaded
    return data


def setup():
    gen_dataset(src, dst_tr, dst_ts, dst_vl)


if __name__ == "__main__":
    setup()
