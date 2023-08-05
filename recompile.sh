#! /bin/bash

cd ./lib/pg/nsg
rm -rf build
mkdir build 
cd build 
cmake .. 
make
cd ../..
make
python wrapper2.py