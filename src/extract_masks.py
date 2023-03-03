import argparse
from PIL import Image
import os
import numpy as np
import shutil
import yaml
import tqdm
from scipy import ndimage as spndi
import scipy


class Palette:
    def __init__(self, yml, color):
        self.max_r = yml["input"][color]["R"]["max"]
        self.min_r = yml["input"][color]["R"]["min"]

        self.max_g = yml["input"][color]["G"]["max"]
        self.min_g = yml["input"][color]["G"]["min"]

        self.max_b = yml["input"][color]["B"]["max"]
        self.min_b = yml["input"][color]["B"]["min"]

        self.palette = np.array(yml["output"]["blue_and_red"])


def extract_mask(limg, uimg, palette):
    h, w, _ = uimg.shape
    mask = np.ones((h, w), dtype=bool)
    mask = mask and limg[0] > palette.r.min and limg[0] < palette.r.max
    mask *= mask and limg[1] > palette.g.min and limg[1] < palette.g.max
    mask *= mask and limg[2] > palette.b.min and limg[2] < palette.b.max
    mask = spndi.binary_dilation(mask, iterations=2)
    mask = spndi.binary_fill_holes(mask)
    mask = spndi.binary_erosion(mask, iterations=2)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled", type=str, required=True)
    parser.add_argument("--labeled", type=str, required=True)
    parser.add_argument("--config", type=str, default="green")
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    output_dir = args.output
    output_dir = os.path.join(output_dir, args.config)

    labeled_dir = args.labeled
    labeled_dir = os.path.join(labeled_dir, args.config)

    unlabeled_dir = args.unlabeled
    unlabeled_dir = os.path.join(unlabeled_dir, args.config)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    labeled = os.listdir(labeled_dir)
    unlabeled = os.listdir(unlabeled_dir)

    labeled = [file for file in labeled if file.endswith(".jpg")]
    unlabeled = [file for file in unlabeled if file.endswith(".jpg")]

    ypath = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(ypath, "r") as f:
        yml = yaml.safe_load(f)
    palette = Palette(yml, args.config)

    for file in tqdm.tqdm(labeled):
        lpath = os.path.join(labeled_dir, file)
        upath = os.path.join(unlabeled_dir, file)

        if not os.path.exists(upath):
            print(f"Unlabeled image for {file} does not exist")
            continue

        limg = np.array(Image.open(lpath))
        uimg = np.array(Image.open(upath))

        mask = extract_mask(limg, uimg, Palette)
        png = Image.fromarray(mask).convert("P")
        png.putpalette(palette.palette)

        file = file.replace(".jpg", ".png")
        file = os.path.join(output_dir, file)
        png.save(file)
