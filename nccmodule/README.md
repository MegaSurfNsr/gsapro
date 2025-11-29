Requirements:

c++ libraries:
OpenCV 3.2.0
CUDA 11.8
pybind11 https://github.com/pybind/pybind11.git
Python3
libtorch-cxx11-abi-shared-with-deps-2.0.0+cu117 (cxx11 ABI) https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu117.zip

python packages:
python (3.8.16)
torch ('1.13.1+cu117')
matplotlib ('3.5.3')
<!-- opencv-python ('4.6.0') -->

Usage：
change the cmakelist according to your enviroment settings.

run c++ version:
1. rename cxxCMakeList.txt to CMakeList.txt
2. build
3. ./ACMH ${dense_folder}

run python version:
1. rename bindingCMakeList.txt to CMakeList.txt
2. pip install .
3. import ncc
