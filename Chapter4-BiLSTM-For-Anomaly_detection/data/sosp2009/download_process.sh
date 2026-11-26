#!/bin/bash
# Downloads the SOSP 2009 HDFS dataset into <chapter4>/datasets/sosp2009/
set -e
# resolve paths from this script's location so it works from any directory
DEST="$(cd "$(dirname "$0")/../.." && pwd)/datasets/sosp2009"
mkdir -p "$DEST"
cd "$DEST"
curl -fL --retry 3 -o HDFS_1.tar.gz "https://zenodo.org/records/3227177/files/HDFS_1.tar.gz?download=1"
tar -zxvf HDFS_1.tar.gz
rm HDFS_1.tar.gz
# newer archive versions keep the label file under preprocessed/
if [ ! -f anomaly_label.csv ] && [ -f preprocessed/anomaly_label.csv ]; then
    cp preprocessed/anomaly_label.csv .
fi
