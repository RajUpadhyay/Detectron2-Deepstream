/**
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 * 
 * Edited by: https://github.com/RajUpadhyay/Detectron2-Deepstream
 */


#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

// This is just the function prototype. The definition is written at the end of the file.
extern "C" bool NvDsInferParseCustomDetectron2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList);

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomDetectron2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {

    static const uint kNUM_CLASSES_DETECTRON2 = 80;
    if (kNUM_CLASSES_DETECTRON2 != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << kNUM_CLASSES_DETECTRON2 << std::endl;
    }

    if (outputLayersInfo.empty()) {
        std::cerr << "output layer is empty" << std::endl;;
        return false;
    }

    int *num_buffer = (int *)outputLayersInfo[0].buffer;
    float *bboxes_buffer = (float *)outputLayersInfo[1].buffer;
    float *scores_buffer = (float *)outputLayersInfo[2].buffer;
    int *classes_buffer = (int *)outputLayersInfo[3].buffer;
    float *mask_buffer = (float *)outputLayersInfo[4].buffer;

    uint w = networkInfo.width;
    uint h = networkInfo.height;

    uint mask_w = outputLayersInfo[4].inferDims.d[2];
    uint mask_h = outputLayersInfo[4].inferDims.d[1];
    uint mask_size = mask_w * mask_h;

    float* det;
    float* data;

    for (int i = 0; i < num_buffer[0]; i++) {
        det = bboxes_buffer + i * 4;
        data = mask_buffer + i * mask_w * mask_h;

        if (scores_buffer[i] < 0.8)
            continue;

        if (classes_buffer[i] != 0)
            continue;

        NvDsInferInstanceMaskInfo object;
        object.classId = classes_buffer[i];
        object.detectionConfidence = scores_buffer[i];

        object.left = CLIP(det[0] * w, 0, w - 1);
        object.top = CLIP(det[1] * h, 0, h - 1);
        object.width = CLIP((det[2] - det[0]) * w, 0, w - 1);
        object.height = CLIP((det[3] - det[1]) * h, 0, h - 1);

        object.mask_width = mask_w;
        object.mask_height = mask_h;
        object.mask_size = sizeof(float) * mask_size;
        object.mask = new float[mask_w * mask_h];

        memcpy(object.mask, data, sizeof(float) * mask_h * mask_w);

        objectList.push_back(object);
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDetectron2);
