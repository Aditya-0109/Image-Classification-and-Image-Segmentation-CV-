import os
from PIL import Image
from dataset_class import Custom_IDD_dataset, tf_img, tf_mask, colour_map
from gen_res import get_res
import shutil
from tqdm import tqdm
import numpy as np


def split_gen(
    source="archive/val",
    root="cityscape",
    img_dir="image_archive",
    mask_dir="mask_archive",
):

    os.makedirs(os.path.join(root, img_dir), exist_ok=True)
    os.makedirs(os.path.join(root, mask_dir), exist_ok=True)

    for i, img_file in enumerate(os.listdir(source)):
        img_file = os.path.join(source, img_file)
        tot_img = Image.open(img_file)
        wdt, ht = tot_img.size

        lhf = (0, 0, wdt // 2, ht)
        rgt = (wdt // 2, 0, wdt, ht)

        image = tot_img.crop(lhf)
        mask = tot_img.crop(rgt)

        image.save(os.path.join(root, img_dir, f"image_{i}.jpg"))
        mask.save(os.path.join(root, mask_dir, f"mask_{i}.jpg"))


def gen_label_masks(colored_dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    itrs = range(len(colored_dataset))
    for i in tqdm(itrs, desc="generating label masks:"):
        _, colored_mask = colored_dataset[i]
        colored_mask_array = np.asarray(colored_mask)

        label_mask = np.zeros(
            (colored_mask_array.shape[0], colored_mask_array.shape[1]), dtype=np.uint8
        )
        for label_value, color in colour_map.items():
            label_mask[np.all(colored_mask_array == color, axis=-1)] = label_value

        label_mask_img = Image.fromarray(label_mask)
        label_mask_img.save(os.path.join(output_dir, f"mask_{i}.jpg"))

    print(f"label masks saved at {output_dir}")
    return output_dir


def setup(data_gen=False):
    if data_gen:
        split_gen()
    city_data = Custom_IDD_dataset(
        "cityscape", transform=tf_img, target_transform=tf_mask
    )
    if data_gen:
        res_data = get_res(city_data, "cityscape")
    else:
        res_data = Custom_IDD_dataset(
            "cityscape",
            mask_dir="res_data",
            transform=tf_img,
            target_transform=tf_mask,
        )
    if data_gen:
        gen_label_masks(city_data, "cityscape/label_mask")
    test_data = Custom_IDD_dataset(
        "cityscape",
        mask_dir="label_mask",
        transform=tf_img,
        target_transform=tf_mask,
    )
    return res_data, test_data
