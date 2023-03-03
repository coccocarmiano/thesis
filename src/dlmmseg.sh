#!/bin/bash

if [ -d mmsegmentation ]; then
    echo "mmsegmentation repository already present"
else
    git clone $(cat .mmlink)
fi
