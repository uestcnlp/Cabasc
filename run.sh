#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

python cmain.py -m mlpattv1_tri_single -d lap

python cmain.py -m mlpattv1_tri_single -d res

python cmain.py -m mlpattv1_tri_single -d dong

