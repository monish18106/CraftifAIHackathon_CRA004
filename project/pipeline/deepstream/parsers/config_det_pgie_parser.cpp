// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[ParseConfigDetPgie] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape (batch stripped): [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // 84 = 4 box coords (cx,cy,w,h) + 80 class scores
    // Need to transpose from [84, 8400] to iterate per detection.

    const unsigned int featureDim = layer0.inferDims.d[0]; // 84
    const unsigned int numDetections = layer0.inferDims.d[1]; // 8400

    // Validate dimensions
    if (featureDim != 4u + numClasses) {
        // Try the other orientation: [8400, 84]
        // We handle both just in case, but expected is [84, 8400].
        fprintf(stderr, "[ParseConfigDetPgie] unexpected featureDim=%u vs 4+numClasses=%u\n",
                featureDim, 4 + numClasses);
        return false;
    }

    // data layout: data[channel * numDetections + det_idx] for [84, 8400]
    // Channels 0..3 = cx, cy, w, h (in network-input pixel coords)
    // Channels 4..83 = class scores (post-sigmoid probabilities)

    // We only keep class 0 (person) per the job spec, but we still do argmax
    // across all 80 classes so we don't mis-label a non-person as person.

    for (unsigned int i = 0; i < numDetections; ++i) {
        // Find best class
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float s = data[(4u + c) * numDetections + i];
            if (s > maxScore) {
                maxScore = s;
                bestClass = c;
            }
        }

        // Only keep person (class 0)
        if (bestClass != 0) continue;

        // Threshold
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.35f;
        if (maxScore < thr) continue;

        // Box coords in network-input pixel space (already absolute, not normalized)
        float cx = data[0u * numDetections + i];
        float cy = data[1u * numDetections + i];
        float bw = data[2u * numDetections + i];
        float bh = data[3u * numDetections + i];

        // Convert centre-size to top-left corner + size
        float left = cx - bw * 0.5f;
        float top  = cy - bh * 0.5f;

        // Clamp to network input
        left = std::max(0.0f, std::min(left, netW));
        top  = std::max(0.0f, std::min(top,  netH));
        bw   = std::max(0.0f, std::min(bw, netW - left));
        bh   = std::max(0.0f, std::min(bh, netH - top));

        // Skip degenerate boxes
        if (bw <= 1.0f || bh <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = bw;
        obj.height = bh;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}