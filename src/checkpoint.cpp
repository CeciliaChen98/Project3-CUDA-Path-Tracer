#include "checkpoint.h"
#include "scene.h"
#include "sceneStructs.h"

#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>

using namespace ckpt;

// Quantize floats so tiny noise won’t flip the signature.
static inline float qf(float v, float eps = 1e-6f) {
    return std::round(v / eps) * eps;
}

// ---- tiny IO helpers ----
struct ChunkHdr { char tag[4]; uint64_t byteSize; };
static bool writeAll(std::ofstream& f, const void* p, size_t n) { f.write((const char*)p, n); return f.good(); }
static bool readAll(std::ifstream& f, void* p, size_t n) { f.read((char*)p, n); return f.good(); }
static bool writeChunk(std::ofstream& f, const char tag[4], const void* data, uint64_t bytes) {
    ChunkHdr ch; std::memcpy(ch.tag, tag, 4); ch.byteSize = bytes;
    if (!writeAll(f, &ch, sizeof(ch))) return false;
    if (bytes == 0) return true;
    return writeAll(f, data, bytes);
}

bool ckpt::saveCheckpoint(const Scene& scene, const std::string& filename)
{
    const RenderState& s = scene.state;
    const uint32_t W = s.camera.resolution.x;
    const uint32_t H = s.camera.resolution.y;

    // Stage film
    std::vector<glm::vec3> image(s.image.size());
    std::memcpy(image.data(), s.image.data(), image.size() * sizeof(glm::vec3));

    std::string tmp = filename + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) { std::cerr << "saveCheckpoint: cannot open " << tmp << "\n"; return false; }

    Header hdr{};
    hdr.magic = 0x5054434B; // "PTCK"
    hdr.version = 4;
    hdr.width = W;
    hdr.height = H;
    hdr.iteration_done = s.iterations;   
    hdr.traceDepth = s.traceDepth;   

    if (!writeAll(out, &hdr, sizeof(hdr))) return false;
    if (!writeChunk(out, "ACCU", image.data(), uint64_t(image.size()) * sizeof(glm::vec3))) return false;

    out.flush(); out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, filename, ec);
    if (ec) { std::cerr << "rename failed: " << ec.message() << "\n"; return false; }
    return true;
}

bool ckpt::loadCheckpoint(Scene& scene, const std::string& filename)
{

    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "loadCheckpoint: cannot open " << filename << "\n"; return false; }

    Header hdr{};
    if (!readAll(in, &hdr, sizeof(hdr))) return false;
    if (hdr.magic != 0x5054434B) { std::cerr << "Bad magic\n"; return false; }
    if (hdr.version != 4) { std::cerr << "Unsupported version\n"; return false; }

    // Restore basic state
    RenderState& s = scene.state;
    s.camera.resolution.x = int(hdr.width);
    s.camera.resolution.x = int(hdr.height);
    s.traceDepth = hdr.traceDepth;                              
    s.iterations = hdr.iteration_done;

    // Read chunks
    std::vector<glm::vec3> image;
    image.resize(size_t(hdr.width) * hdr.height);

    while (in.good()) {
        ChunkHdr ch{};
        if (!readAll(in, &ch, sizeof(ch))) break;

        if (std::memcmp(ch.tag, "ACCU", 4) == 0) {
            if (ch.byteSize != image.size() * sizeof(glm::vec3)) { std::cerr << "ACCU size mismatch\n"; return false; }
            if (!readAll(in, image.data(), ch.byteSize)) return false;
        }
        else {
            // Unknown chunk -> skip
            std::vector<char> skip(ch.byteSize);
            if (!readAll(in, skip.data(), ch.byteSize)) return false;
        }
    }

    s.image.resize(image.size());
    std::memcpy(s.image.data(), image.data(), image.size() * sizeof(glm::vec3));
    return true;
}
