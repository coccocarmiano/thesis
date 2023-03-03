from matplotlib import pyplot as plt
import tqdm
from PIL import Image
import numpy as np
import os
from os import path as osp
from scipy import ndimage as ndi

SRC_MASKS_DIR = "../data/hyper/dataset/masks"
DST_MASKS_DIR = "../data/hyper/dataset/masks-converted"
PALETTE = [20, 20, 240, 20, 240, 20]

if __name__ == "__main__":
    l = os.listdir(SRC_MASKS_DIR)

    for file in tqdm.tqdm(l):
        p = osp.join(SRC_MASKS_DIR, file)
        img = np.array(Image.open(p).convert("L"))
        img[img < 230] = 0
        img = Image.fromarray(img).convert("P")
        img.putpalette(PALETTE)
        file = file.replace(".jpg", ".png")
        img.save(osp.join(DST_MASKS_DIR, file))
