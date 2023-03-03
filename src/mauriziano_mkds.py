import shutil
import tqdm
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import scipy as sp
from os import path as osp
from PIL import Image

DS_ROOT = "../data/mauri/dataset"
SANE_DIR = osp.join(DS_ROOT, "sane_images")
SICK_LABELED_DIR = osp.join(DS_ROOT, "segmented_images_labeled")
SICK_UNLABELED_DIR = osp.join(DS_ROOT, "segmented_images_unlabeled")
OUTDIR = "out"
PALETTE = [20, 20, 240, 20, 240, 20]


def cut(img):
    _, w, _ = img.shape
    img = img[:, :w * 2 // 3, :]
    img = img[45:-55, 40:-10, :]
    return img


def clean_dirs():
    cwd = osp.dirname(__file__)
    out = osp.join(cwd, "..", "data", "mauri", OUTDIR)
    if osp.exists(out):
        shutil.rmtree(out)
    os.mkdir(out)
    return out


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

    return mask


if __name__ == "__main__":
    sane_files = os.listdir(SANE_DIR)
    sick_labeled_files = os.listdir(SICK_LABELED_DIR)
    sick_unlabeled_files = os.listdir(SICK_UNLABELED_DIR)

    sane_files = [file for file in sane_files if file.endswith(".jpg")]
    sick_labeled_files = [
        file for file in sick_labeled_files if file.endswith(".jpg")
    ]
    sick_unlabeled_files = [
        file for file in sick_unlabeled_files if file.endswith(".jpg")
    ]

    output_dir = clean_dirs()

    print("Sane Images:", len(sane_files))
    print("Sick Labeled Images:", len(sick_labeled_files))
    print("Sick Unlabeled Images:", len(sick_unlabeled_files))

    cnt = 0

    for file in tqdm.tqdm(sick_labeled_files):
        if file not in sick_unlabeled_files:
            print(f"File {file} is in labeled but not in unlabeled")
            continue
        t1 = osp.join(SICK_LABELED_DIR, file)
        t2 = osp.join(SICK_UNLABELED_DIR, file)

        lsick = cut(np.array(Image.open(t1)))
        usick = cut(np.array(Image.open(t2)))
        mask = outline_mask(lsick)
        mask = mask.astype(np.uint8)

        mask = Image.fromarray(mask).convert("P")
        mask.putpalette(PALETTE)

        img = Image.fromarray(usick)

        mask_out = osp.join(output_dir, f"{cnt}.png")
        img_out = osp.join(output_dir, f"{cnt}.jpg")

        mask.save(mask_out)
        img.save(img_out)

        cnt += 1

    # cnt = 0
    # blank = cut(np.array(Image.open(osp.join(SANE_DIR, sane_files[0]))))
    # h, w = blank.shape[:2]
    # blank = np.zeros((h, w, 3), dtype=np.uint8)
    # blank = Image.fromarray(blank).convert("P")
    # blank.putpalette(PALETTE)

    # for file in tqdm.tqdm(sane_files):
    # inpath = osp.join(SANE_DIR, file)
    # img = cut(np.array(Image.open(inpath)))
    # img = Image.fromarray(img)

    # img.save(osp.join(output_dir, f"{cnt}-sane.jpg"))
    # blank.save(osp.join(output_dir, f"{cnt}-sane.png"))

    # cnt += 1
