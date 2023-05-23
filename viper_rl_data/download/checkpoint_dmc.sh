#!/bin/sh

# mkdir -p $1
# cd $1
FILE_PATH=$(realpath "$0")
DOWNLOAD_DIR=$(dirname "$FILE_PATH")
DATA_DIR=$(dirname "$DOWNLOAD_DIR")
CHECKPOINT_DIR=${DATA_DIR}/checkpoints
echo "Saving checkpoint to:"
echo $CHECKPOINT_DIR

mkdir -p $CHECKPOINT_DIR
cd $CHECKPOINT_DIR

for i in aa
do
    ia download dmc_checkpoint_$i dmc.tar.part$i
    mv dmc_checkpoint_$i/dmc.tar.part$i .
    rmdir dmc_checkpoint_$i
done

cat dmc.tar.part* | tar x

rm dmc.tar.part*