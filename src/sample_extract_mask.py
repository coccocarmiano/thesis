import os
import os.path as osp
import PIL.Image as Image
import numpy
import matplotlib.pyplot as plt
import numpy as np
from mauriziano_mkds import cut, outline_mask

src_labeled = os.listdir("../data/mauri/labeled")[6]
src_unlabeled = src_labeled.replace("labeled", "unlabeled")

im = Image.open(osp.join("../data/mauri/labeled", src_labeled))
im = numpy.array(im)
im = cut(im)
mask = outline_mask(im)

unlabeled = Image.open(osp.join("../data/mauri/unlabeled", src_unlabeled))
unlabeled = numpy.array(unlabeled)
unlabeled = cut(unlabeled)

plt.subplot(2, 2, 1)
plt.imshow(im)
plt.axis("off")

plt.subplot(2, 2, 2)
h, w, = mask.shape
mask_img = np.zeros((h, w, 3), dtype=np.uint8)
mask_img[mask] = [190, 30, 30]
mask_img[np.logical_not(mask)] = [10, 10, 120]
plt.imshow(mask_img)
plt.axis("off")
plt.tight_layout()


plt.subplot(2, 2, 3)
plt.imshow(unlabeled)
plt.axis("off")

cutout = unlabeled.copy()
cutout[np.logical_not(mask)] = [0, 0, 0]
plt.subplot(2, 2, 4)
plt.imshow(cutout)
plt.axis("off")


plt.show()
