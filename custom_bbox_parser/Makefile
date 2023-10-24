################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Edited by: https://github.com/RajUpadhyay/Detectron2-Deepstream
#
################################################################################

CUDA_VER:=11.4
SRCFILES:=nvdsparsebbox_detectron2.cpp
TARGET_LIB:=libnvdsinfer_custom_bbox_detectron2.so
DEEPSTREAM_PATH:=/home/tmc/frcnn_deepstream/

ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif
CC:= g++

CFLAGS:= -Wall -Werror -std=c++11 -shared -fPIC
CFLAGS+= -I$(DEEPSTREAM_PATH)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include -I/opt/nvidia/deepstream/deepstream-6.2/sources/includes/

LIBS:= -lnvinfer -lnvparsers -L/usr/local/cuda-$(CUDA_VER)/lib64
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
