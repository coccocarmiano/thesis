import os
import os.path as osp
import matplotlib.pyplot as plt
import PIL.Image as Image
from mauriziano_mkds import cut
import numpy as np

f = os.listdir("../data/mauri/labeled")[5]
path_unlabel = osp.join("../data/mauri/unlabeled", f)
path_labeled = osp.join("../data/mauri/labeled", f)

img_unlabel = Image.open(path_unlabel)
img_labeled = Image.open(path_labeled)

img_unlabel = np.array(img_unlabel)
img_labeled = np.array(img_labeled)

img_unlabel = cut(img_unlabel)
img_labeled = cut(img_labeled)

plt.subplot(1, 2, 1)
plt.imshow(img_unlabel)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(img_labeled)
plt.tight_layout()
plt.axis("off")
plt.show()
