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
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomDetectron2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {

    static const uint kNUM_CLASSES_DETECTRON2 = 1;
    if (kNUM_CLASSES_DETECTRON2 != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << kNUM_CLASSES_DETECTRON2 << std::endl;
    }

    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }

    int *num_buffer = (int *)outputLayersInfo[0].buffer;
    float *bboxes_buffer = (float *)outputLayersInfo[1].buffer;
    float *scores_buffer = (float *)outputLayersInfo[2].buffer;
    int *classes_buffer = (int *)outputLayersInfo[3].buffer;

    uint w = networkInfo.width;
    uint h = networkInfo.height;

    float* det;

    for (int i = 0; i < num_buffer[0]; i++) {
        det = bboxes_buffer + i * 4;

        NvDsInferObjectDetectionInfo object;
        object.classId = classes_buffer[i];
        object.detectionConfidence = scores_buffer[i];

        object.left = CLIP(det[0] * w, 0, w - 1);
        object.top = CLIP(det[1] * h, 0, h - 1);
        object.width = CLIP((det[2] - det[0]) * w, 0, w - 1);
        object.height = CLIP((det[3] - det[1]) * h, 0, h - 1);

        objectList.push_back(object);
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDetectron2);
