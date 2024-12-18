# ONNX Controller

This document provides instructions on how to use the ONNX runtime controller in the TOCABI project.

## Installation 

### Launch install_onnx.sh
Installing prebuilt package(CPU) on ```/usr/local/lib``` and ```/usr/local/include```
```sh
sudo ./install_onnx.sh
```        
### Helpful Documents
- [ONNX Runtime C++ 개발환경 설정 (Linux)](https://madplayer.github.io/development-environment/)
- [ONNX Github Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)

### How to export Onnx Files?
1. rsl_rl
   
   [Isaac Lab repo](https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/utils/wrappers/rsl_rl/exporter.py#L94) contains onnx exporter for a2c algorithm

2. rl_games
   
   [Tutorial for exporting on rl_games](https://www.tylerbarkin.com/isaac-gym-to-onnx)
