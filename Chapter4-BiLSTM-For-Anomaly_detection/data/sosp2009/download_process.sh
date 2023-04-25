mkdir ../../datasets/
mkdir ../../datasets/sosp2009/
wget https://zenodo.org/record/3227177/files/HDFS_1.tar.gz -o ../../datasets/sosp2009/HDFS_1.tar.gz
# shellcheck disable=SC2164
cd ../../datasets/sosp2009/
tar -zxvf HDFS_1.tar.gz
rm HDFS_1.tar.gz