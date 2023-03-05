#!/bin/bash

CONF_PATH=confs/mgi_dataset.py
CKPT_NAME="iter_1750.pth"
CKPT_PATH="$(realpath model)/$CKPT_NAME"
MODEL_STORE="$(realpath model)"
MODEL_NAME="mgi"
MODEL_SRC="$MODEL_STORE/$MODEL_NAME"

echo "Checkpoint Path: $CKPT_PATH"
echo "Model Store: $MODEL_STORE"
echo "Model Name: $MODEL_NAME"
echo "Model Path: $MODEL_SRC"


python fix_model.py

# Convert Model
python mmsegmentation/tools/torchserve/mmseg2torchserve.py -f $CONF_PATH $CKPT_PATH --output-folder $MODEL_STORE --model-name ${MODEL_NAME}

# Set the docker image
docker build -t mmseg-serve:latest docker/serve


# Run the model
docker run --rm --cpus 8 -p8080:8080 -p8081:8081 -p8082:8082 \
    --mount type=bind,source=${MODEL_STORE},target=/home/model-server/model-store\
    mmseg-serve:latest

