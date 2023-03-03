_base_ = "../mmsegmentation/configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py"
num_classes = 1
CLASSES = ("sane", "sick")
PALETTE = [
    [40, 220, 40],
    [220, 40, 40],
]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
norm_cfg = dict(type="BN", requires_grad=True)
dataset_type = "MGIDataset"
gpu_ids = [0]
seed = 0
device = "cuda"
model = dict(
    decode_head=dict(
        norm_cfg=norm_cfg,
        loss_decode=dict(type="CrossEntropyLoss",
                         use_sigmoid=True,
                         loss_weight=1.0),
        out_channels=num_classes,
        num_classes=num_classes,
    ),
    auxiliary_head=[
        dict(
            type="FCNHead",
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=num_classes,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            out_channels=1,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=0.4,
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=num_classes,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            out_channels=1,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=0.4,
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=num_classes,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            out_channels=1,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=0.4,
            ),
        ),
        dict(
            type="FCNHead",
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=num_classes,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            out_channels=1,
            loss_decode=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=0.4,
            ),
        ),
    ],
)

work_dir = "test_output"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", img_scale=(800, 800), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=(600, 600), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    # dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(800, 800),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type="MGIDataset",
        data_root="../data/hyper/dataset",
        img_dir="images",
        ann_dir="masks-converted",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split="splits/train.txt",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="MGIDataset",
        data_root="../data/mauri/",
        img_dir="out/",
        ann_dir="out/",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split="splits/val.txt",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="MGIDataset",
        data_root="../data/mauri/",
        img_dir="out/",
        ann_dir="out/",
        img_suffix=".jpg",
        seg_map_suffix=".png",
        split="splits/test.txt",
        pipeline=test_pipeline,
    ),
)

checkpoint_config = dict(
    interval=2000,
    meta=dict(
        CLASSES=CLASSES,
        PALETTE=PALETTE,
    ))
evaluation = dict(
    interval=2000
)
runner = dict(
    max_iters=160_000
)
log_config = dict(
    interval=2000
)
evaluation = dict(
    interval=2000
)
checkpoint_config = dict(
    interval=400
)
