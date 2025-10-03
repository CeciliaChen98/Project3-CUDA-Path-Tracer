#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 sampleUniformDiskConcentric(glm::vec2 xi) {

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

__host__ __device__ glm::vec3 sampleCosineHemisphere(glm::vec2 xi) {

    glm::vec3 diskSample = sampleUniformDiskConcentric(xi);
    float z = sqrt(glm::max(0.0f, 1.0f - diskSample.x * diskSample.x - diskSample.y * diskSample.y));
    return glm::vec3(diskSample.x, diskSample.y, z);
}

__host__ __device__ float cosHemispherePDF(float cosTheta) {
    return (cosTheta > 0.f) ? cosTheta * INV_PI : 0.f;
}

__host__ __device__ float AbsDot(glm::vec3 a, glm::vec3 b) {
    return abs(dot(a, b));
}

__host__ __device__ glm::vec3 Faceforward(glm::vec3 n, glm::vec3 v) {
    return (dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__ glm::mat3 LocalToWorld(const glm::vec3& nor)
{
    glm::vec3 tan, bit;
    if (abs(nor.x) > abs(nor.y)) {
        tan = glm::vec3(-nor[2], 0, nor[0]) / sqrt(nor[0] * nor[0] + nor[2] * nor[2]);
    }
    else {
        tan = glm::vec3(0, nor[2], -nor[1]) / sqrt(nor[1] * nor[1] + nor[2] * nor[2]);
    }
    bit = glm::normalize(glm::cross(tan, nor));
    return glm::mat3(tan, bit, nor);
}


__host__ __device__ float FresnelSchlick(float cosThetaI, float ior) {
    float R = glm::pow((1.0f - ior) / (ior + 1.0f), 2.0f);
    return R + (1.0f - R) * glm::pow((1.0f - cosThetaI), 5.0f);
}


__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng){

    glm::vec3 wi;
    float pdf;
    glm::vec3 wo = normalize(pathSegment.ray.direction);
    thrust::uniform_real_distribution<float> u01(0, 1);
    if (m.type == DIFFUSE) {
        glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
        wi = sampleCosineHemisphere(xi);
        pdf = cosHemispherePDF(wi.z);

        if (pdf <= 0.0f) {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
            return;
        }
        glm::mat3 TBN = LocalToWorld(normal);
        wi = normalize(TBN * wi);  // transform to world
        pathSegment.color *= m.color * INV_PI * AbsDot(wi, normal) / pdf;
        pathSegment.ray.origin = intersect + EPSILON * normal;
    }
    else if (m.type == SPECULAR) {
        wi = glm::reflect(wo, normal);
        pathSegment.ray.origin = intersect + EPSILON * normal;
        pathSegment.color *= m.color;
    }
    else if (m.type == TRANS) {

        bool entering = glm::dot(wo, normal) > 0.0f;
        glm::vec3 n = entering ? -1.0f * normal : normal;
        float etaA = 1.0f;
        float etaB = (m.indexOfRefraction > 0.0f) ? m.indexOfRefraction : 1.55f;
        float eta = entering ? (etaB / etaA) : (etaA / etaB);

        float xi = u01(rng);
        float cosTheta = glm::dot(normalize(-wo), normalize(normal));
        float prob = FresnelSchlick(cosTheta, etaB);

        wi = glm::refract(wo, n, eta);
        pathSegment.color *= m.color;
        if (glm::length(wi) < 0.01f) {
            wi = glm::reflect(wo, normal);
        }
        if (prob > m.hasRefractive) {
            wi = glm::reflect(wo, normal);
        }
        pathSegment.ray.origin = intersect + 0.001f * wi;
    }
    else if (m.type == GLASS) {
        bool entering = glm::dot(wo, normal) > 0.0f;
        glm::vec3 n = entering ? -1.0f * normal : normal;
        float etaA = 1.0f;
        float etaB = (m.indexOfRefraction > 0.0f) ? m.indexOfRefraction : 1.55f;
        float eta = entering ? (etaB / etaA) : (etaA / etaB);

        float xi = u01(rng);
        float cosTheta = glm::dot(-wo, normal);
        float prob = FresnelSchlick(cosTheta, etaB);

        if (0.5f < xi) {
            wi = glm::reflect(wo, normal);
        }
        else {
            wi = glm::refract(wo, n, eta);
            if (glm::length(wi) < 0.01f) {
                wi = glm::reflect(wo, normal);
            }
            if (prob > m.hasRefractive) {
                wi = glm::reflect(wo, normal);
            }
        }
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + 0.001f * wi;
    }
    pathSegment.ray.direction = normalize(wi);
}
