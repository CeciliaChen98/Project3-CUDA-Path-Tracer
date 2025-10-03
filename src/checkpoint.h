#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <functional>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "sceneStructs.h"  // Camera/RenderState/Material/Geom/PathSegment etc.
class Scene;

namespace ckpt {

    struct Header {
        uint32_t magic;
        uint32_t version;
        uint32_t width, height;
        uint32_t iteration_done;
        int32_t  traceDepth;  
    };

    std::string makeSceneSummaryString(const Scene& scene);

    bool saveCheckpoint(const Scene& scene,
        const std::string& filename);

    bool loadCheckpoint(Scene& scene,
        const std::string& filename);

} // namespace ckpt
