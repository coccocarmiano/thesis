import torch
import mmcv

from mmseg.apis import train_segmentor, init_segmentor
# from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset

from mmcv import Config
from mmcv.utils import build_from_cfg

from os import path as osp

classes = ("sane", "sick")
PALETTE = [[20, 20, 220], [20, 220, 20]]


@DATASETS.register_module()
class MGIDataset(CustomDataset):
    CALSSES = classes
    PALETTE = PALETTE

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)


cfg = Config.fromfile("./confs/mgi_dataset.py")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
cfg.device = dev

train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)
test_dataset = build_dataset(cfg.data.test)

# model = build_segmentor(cfg.model).to(dev)
model = init_segmentor(cfg)
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

print("Train dataset - Number of Samples:", len(train_dataset))
print("Test dataset - Number of Samples:", len(test_dataset))
print("Val dataset - Number of Samples:", len(val_dataset))

train_segmentor(model, [train_dataset], cfg, distributed=False, validate=True)