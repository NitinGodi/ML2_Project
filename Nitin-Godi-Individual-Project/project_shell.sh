#!/bin/sh
pip install -q kaggle
mkdir -p ~/.kaggle
cp $1/kaggle.json ~/.kaggle/
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d bittlingmayer/amazonreviews -p $2
cd $2/Code
unzip amazonreviews.zip
bzip2 -dk train.ft.txt.bz2
bzip2 -dk test.ft.txt.bz2
rm amazonreviews.zip train.ft.txt.bz2 test.ft.txt.bz2 