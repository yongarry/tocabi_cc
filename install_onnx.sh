#!/bin/bash

echo "Dyros ONNX Auto Installer"

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

echo "Starting Install prebuilt onnxruntime"
mkdir Temp
cd Temp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xvzf onnxruntime-linux-x64-1.20.1.tgz
cd onnxruntime-linux-x64-1.20.1

echo "Copying files to /usr/local"
sudo cp -r include/ /usr/local/include/onnxruntime
sudo cp -r lib/libonnxruntime.* /usr/local/lib/

cd ../..
rm -rf Temp
sudo ldconfig