#!/bin/bash

DDDIR="./deepdetect/"
DEBDIR="./metadata/deb"

DEBDIR_DEST="deb/"
# Install dependencies for Building

sudo apt-get install --yes autoconf automake libtool-bin build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurlpp-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev cmake libgoogle-perftools-dev


### Build steps ###
olddir=$(pwd)
cd $DDDIR
# Download libcurlpp-dev, see
sudo apt-get remove --yes libcurlpp0
git clone https://github.com/datacratic/curlpp
cd curlpp
./autogen.sh
./configure --prefix=/usr --enable-ewarning=no
make
sudo make install

cd $olddir
cd $DDDIR

# Build deepdetect
mkdir build
cd build
cmake ..
make



## End build steps ###

cd $olddir

# Build package
mkdir $DDDIR$DEBDIR_DEST
cp -r $DEBDIR/* $DDDIR$DEBDIR_DEST

#Get version number using this package commit date
cd $DDDIR
package_version=$(date +%Y%m%d%H%M)
cd -