import shutil
import tqdm
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy as sp
from os import path as osp
from PIL import Image

PALETTE = [20, 20, 240, 20, 240, 20]


def cut(img):
    _, w, _ = img.shape
    img = img[:, :w * 2 // 3, :]
    img = img[45:-55, 40:-10, :]
    return img


def outline_mask(img):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)

    red = img[:, :, 0]
    green = img[:, :, 1]

    mask *= red < 200
    mask *= red > 170

    mask *= green < 235
    mask *= green > 220

    mask = sp.ndimage.binary_dilation(
        mask, iterations=2)  # Some paths are not perfectly closed
    mask = sp.ndimage.binary_fill_holes(mask)  # Close the holes
    mask = sp.ndimage.binary_erosion(mask,
                                     iterations=5)  # Remove the green border

    if mask.sum() == 0:
        raise ValueError("No mask found")

    return mask


if __name__ == "__main__":
    data_dir_labeled = "../data/mauri/labeled"
    data_dir_unlabeled = "../data/mauri/unlabeled"
    data_dir_masks = "../data/mauri/masks"
    data_out = "../data/mauri/images"

    if os.path.isdir(data_dir_masks):
        shutil.rmtree(data_dir_masks)
    os.mkdir(data_dir_masks)

    if os.path.isdir(data_out):
        shutil.rmtree(data_out)
    os.mkdir(data_out)

    labeled_files = [file for file in os.listdir(
        data_dir_labeled) if file.endswith(".jpg")]

    unlabeled_files = [file for file in os.listdir(
        data_dir_unlabeled)]
    unlabeled_files = [
        file for file in unlabeled_files if file in labeled_files]

    val_txt = open("../data/mauri/splits/val.txt", "w")
    test_txt = open("../data/mauri/splits/test.txt", "w")
    train_txt = open("../data/mauri/splits/train.txt", "w")
    pick_split = [train_txt, train_txt, val_txt, test_txt]

    for idx, file in enumerate(tqdm.tqdm(labeled_files)):
        if file not in unlabeled_files:
            print(f"File {file} for idx {idx} not found in unlabeled folder")
            continue

        src_path = osp.join(data_dir_labeled, file)
        dst_path = osp.join(data_dir_masks, f"{idx}.png")

        img = np.array(Image.open(src_path))
        img = cut(img)
        try:
            mask = outline_mask(img)
        except ValueError:
            print(f"File {file} has no mask")
            continue
        mask = mask.astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        mask.putpalette(PALETTE)
        mask.save(dst_path)

        src_path = osp.join(data_dir_unlabeled, file)
        img = Image.open(src_path)
        img = cut(np.array(img))
        img = Image.fromarray(img)
        dst_path = osp.join(data_out, f"{idx}.jpg")
        img.save(dst_path)

        f = pick_split[idx % len(pick_split)]
        print(idx, file=f)
