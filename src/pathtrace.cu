#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jx = u01(rng);
        float jy = u01(rng);

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (jx + (float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (jy + (float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];
        pathSegments[path_index].intersectIndex = path_index;

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__device__ glm::vec3 sampleUniformDiskConcentric(glm::vec2 xi) {
    
    float sx = 2.0f * xi[0] - 1.0f;
    float sy = 2.0f * xi[1] - 1.0f;

    float r, theta;
    if (sx == 0 && sy == 0) {
        return glm::vec3(0.0f);
    }
    if (abs(sx) > abs(sy)) {
        r = sx;
        theta = (PI / 4.0f) * (sy / sx);
    }
    else {
        r = sy;
        theta = (PI / 2.0f) - (PI / 4.0f) * (sx / sy);
    }
    return glm::vec3(r * cos(theta), r * sin(theta), 0.0f);
}

__device__ float AbsCosTheta(glm::vec3 w) { return std::abs(w.z); }

__device__ glm::vec3 sampleCosineHemisphere(glm::vec2 xi) {

    glm::vec3 diskSample = sampleUniformDiskConcentric(xi);
    float z = sqrt(glm::max(0.0f, 1.0f - diskSample.x * diskSample.x - diskSample.y * diskSample.y));
    return glm::vec3(diskSample.x, diskSample.y, z);
}

__device__ float cosHemispherePDF(float cosTheta) {
    return (cosTheta > 0.f) ? cosTheta * INV_PI : 0.f;
}

__device__ float AbsDot(glm::vec3 a, glm::vec3 b) {
    return abs(dot(a, b));
}

__device__ glm::vec3 Faceforward(glm::vec3 n, glm::vec3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}

__device__ glm::mat3 LocalToWorld(const glm::vec3& nor)
{
    glm::vec3 tan, bit;
    if (abs(nor.x) > abs(nor.y)) {
        tan = glm::vec3(-nor[2], 0, nor[0]) / sqrt(nor[0] * nor[0] + nor[2] * nor[2]);
    }else {
        tan = glm::vec3(0, nor[2], -nor[1]) / sqrt(nor[1] * nor[1] + nor[2] * nor[2]);
    }
    bit = glm::normalize(glm::cross(tan, nor));
    return glm::mat3(tan, bit, nor);
}

__device__ glm::mat3 WorldToLocal(const glm::vec3& nor) {
    return transpose(LocalToWorld(nor));
}

__device__ glm::vec3 Refract(const glm::vec3& wi, const glm::vec3& n, float eta) {
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = glm::max(0.0f, (1.0f - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1.0f) return glm::vec3(0.0f);
    float cosThetaT = sqrt(1.0f - sin2ThetaT);
    return eta * -wi + (eta * cosThetaI - cosThetaT) * n;
}

__device__ glm::vec3 FresnelDielectricEval(float cosThetaI) {
    // We will hard-code the indices of refraction to be
    // those of glass
    float etaI = 1.0f;
    float etaT = 1.55f;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    float eta = (cosThetaI > 0.0f) ? (etaI / etaT) : (etaT / etaI);
    if (cosThetaI < 0.0) cosThetaI = -cosThetaI;

    float sin2ThetaT = eta * eta * (1.0f - cosThetaI * cosThetaI);

    if (sin2ThetaT >= 1.0) {
        return glm::vec3(1.0); // Fully reflective
    }

    float cosThetaT = sqrt(1.0 - sin2ThetaT);

    // Fresnel reflectance using Schlick's approximation
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return glm::vec3((Rparl * Rparl + Rperp * Rperp) * 0.5f);
}

__device__ glm::vec3 BSDF_reflect(const glm::vec3 & materialColor,const glm::vec3 & wo, glm::vec3& wi,const glm::vec3& n) {
    wi = glm::vec3(-wo.x, -wo.y, wo.z);
    float absCosTheta = AbsCosTheta(wi);
    return materialColor / absCosTheta;
}

__device__ glm::vec3 BSDF_refract(const glm::vec3& materialColor, const glm::vec3& wo, glm::vec3& wi, const glm::vec3& n) {
    float etaA = 1.0f;
    float etaB = 1.55f;

    bool entering = wo.z > 0.0f;
    float eta = entering ? (etaA / etaB) : (etaB / etaA);

    wi = Refract(wo,Faceforward(glm::vec3(0.0f,0.0f,1.0f),wo), eta);
    float absCosTheta = AbsCosTheta(wi);

    if (glm::length(wi) == 0.0f) {
        return glm::vec3(0.0);
    }
    else {
        return materialColor / absCosTheta;
        //return materialColor;
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[pathSegments[idx].intersectIndex];
        Ray ray = pathSegments[idx].ray;
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            glm::vec3 normal = normalize(intersection.surfaceNormal);

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 wi, bsdf;
                float pdf;
                glm::vec3 wo = -ray.direction;
                wo = normalize(WorldToLocal(normal) * wo);

                if (material.type == DIFFUSE) {
                    glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
                    wi = sampleCosineHemisphere(xi);
                    pdf = cosHemispherePDF(wi.z);

                    if (pdf <= 0.0f) {
                        pathSegments[idx].color = glm::vec3(0.0f);
                        pathSegments[idx].remainingBounces = 0;
                        return;
                    }
                    glm::mat3 TBN = LocalToWorld(normal);
                    wi = normalize(TBN * wi);  // transform to world

                    bsdf = materialColor * INV_PI; //albedo
                }
                else if (material.type == SPECULAR) {
                    pdf = 1.0f;
                    bsdf = BSDF_reflect(materialColor, wo, wi, normal);
                    glm::mat3 TBN = LocalToWorld(normal);
                    wi = normalize(TBN * wi);
                }
                else if (material.type == TRANS) {
                    pdf = 1.0f;
                    bsdf = BSDF_refract(materialColor, wo, wi, normal);
                    glm::mat3 TBN = LocalToWorld(normal);
                    wi = normalize(TBN * wi);
                    if (glm::length(wi) == 0.0) {
                        pathSegments[idx].color = glm::vec3(0.0f);
                        pathSegments[idx].remainingBounces = 0;
                        return;
                    }
                }
                else if (material.type == GLASS) {
                    pdf = 1.0f;
                    float random = u01(rng);
                    if (random < 0.5f) {
                        glm::vec3 R = BSDF_reflect(materialColor, wo, wi, normal);
                        bsdf = 2.0f * FresnelDielectricEval(dot(normal, normalize(wi))) * R;
                    }
                    else {
                        glm::vec3 T = BSDF_refract(materialColor, wo, wi, normal);
                        bsdf = 2.0f * (glm::vec3(1.0f) - FresnelDielectricEval(dot(normal, normalize(wi)))) * T;
                    }
                }
                
                pathSegments[idx].color *= bsdf * AbsDot(wi, normal) /pdf;
                glm::vec3 hitPoint = ray.origin + ray.direction * intersection.t;
                pathSegments[idx].ray.origin = hitPoint + EPSILON * normal;
                pathSegments[idx].ray.direction = wi;
                pathSegments[idx].remainingBounces--;
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct IsDeadPath {
    __host__ __device__
        bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;  
    }
};

__global__ void buildMaterialKeys(const ShadeableIntersection* isects,
    const Material* materials,
    int* keys,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    keys[i] = materials[isects[i].materialId].type;
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        thrust::device_vector<int> keys(num_paths);

        // Fill keys on the GPU
        buildMaterialKeys <<<numblocksPathSegmentTracing, blockSize1d >> > (
            dev_intersections, 
            dev_materials,
            thrust::raw_pointer_cast(keys.data()),
            num_paths);
        cudaDeviceSynchronize();  
        
        // Sort dev_paths by keys
        thrust::sort_by_key(
            thrust::device,
            keys.begin(), keys.end(),
            dev_paths
        );

        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        
        auto first = dev_paths;
        auto last = dev_paths + num_paths;
        auto mid = thrust::partition(thrust::device, first, last, IsDeadPath{});

        num_paths = int(mid - first);

        iterationComplete = (num_paths == 0) || (depth >= traceDepth); 
 
        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
