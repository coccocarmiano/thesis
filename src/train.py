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
    CLASSES = classes
    PALETTE = PALETTE

    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)


cfg = Config.fromfile("./confs/mgi_dataset.py")
train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)
test_dataset = build_dataset(cfg.data.test)

model = init_segmentor(cfg, device="cpu")
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

print("Train dataset - Number of Samples:", len(train_dataset))
print("Test dataset - Number of Samples:", len(test_dataset))
print("Val dataset - Number of Samples:", len(val_dataset))

cfg.runner.max_iters = 8000,
cfg.log_config.interval = 2000,
cfg.checkpoint_config.interval = 2000,
cfg.evaluation.interval = 2000

train_segmentor(model, [train_dataset], cfg, distributed=False, validate=True)
