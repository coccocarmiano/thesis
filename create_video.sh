#!/bin/bash

echo "Be sure to execute this script from the project rood dir"

if [ -d data/mauri/video ]; then
    rm -rf data/mauri/video
fi

mkdir data/mauri/video

if [ -d __tmp ]; then
    rm -rf tmp
fi

mkdir .tmp

ls data/mauri/labeled | xargs -I _ cp "data/mauri/labeled/_" .tmp/
ffmpeg -framerate 24 -pattern_type glob -i '.tmp/*.jpg' -c:v libx264 -pix_fmt yuv420p data/mauri/video/labeled.mp4

rm .tmp/*

ls data/mauri/labeled | xargs -I _ cp "data/mauri/unlabeled/_" .tmp/
ffmpeg -framerate 24 -pattern_type glob -i '.tmp/*.jpg' -c:v libx264 -pix_fmt yuv420p data/mauri/video/unlabeled.mp4

rm -rf .tmp
