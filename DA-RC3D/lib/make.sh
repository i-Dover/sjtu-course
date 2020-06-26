#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

#python setup.py build_ext --inplace
#rm -rf build

CUDA_ARCH="-arch= sm_61\
           -gencode=arch=compute_30,code=sm_30 \
           -gencode=arch=compute_35,code=sm_35 \
           -gencode=arch=compute_50,code=sm_50 \
           -gencode=arch=compute_52,code=sm_52 \
           -gencode=arch=compute_60,code=sm_60 \
           -gencode=arch=compute_61,code=sm_61 "

# compile NMS
cd model/nms/src
echo "Compiling nms kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python build.py

# compile roi_temporal_pooling
cd ../../
cd model/roi_temporal_pooling/src
echo "Compiling roi temporal pooling kernels by nvcc..."
nvcc -c -o roi_temporal_pooling_kernel.cu.o roi_temporal_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_61
cd ../
python build.py
