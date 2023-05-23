#!/bin/sh

# mkdir -p $1
# cd $1
FILE_PATH=$(realpath "$0")
DOWNLOAD_DIR=$(dirname "$FILE_PATH")
DATA_DIR=$(dirname "$DOWNLOAD_DIR")
DATASET_DIR=${DATA_DIR}/datasets
echo "Saving dataset to:"
echo $DATASET_DIR

mkdir -p $DATASET_DIR
cd $DATASET_DIR

for i in aa
do
    ia download atari_dataset_$i atari.tar.part$i
    mv atari_dataset_$i/atari.tar.part$i .
    rmdir atari_dataset_$i
done

cat atari.tar.part* | tar x

ia download atari_dataset_mask mask_map.pkl
mv atari_dataset_mask/mask_map.pkl atari
rmdir atari_dataset_mask

rm atari.tar.part*