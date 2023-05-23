#!/bin/sh

# mkdir -p $1
# cd $1
FILE_PATH=$(realpath "$0")
DOWNLOAD_DIR=$(dirname "$FILE_PATH")
DATA_DIR=$(dirname "$DOWNLOAD_DIR")
DATA_DIR=${DATA_DIR}/datasets
echo "Saving dataset to:"
echo $DATA_DIR

mkdir -p $DATA_DIR
cd $DATA_DIR

for i in aa ab ac ad ae
do
    ia download dmc_dataset_$i dmc.tar.part$i
    mv dmc_dataset_$i/dmc.tar.part$i .
    rmdir dmc_dataset_$i
done

cat dmc.tar.part* | tar x

rm dmc.tar.part*