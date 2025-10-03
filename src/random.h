#pragma once
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

// reference: https://hbfs.wordpress.com/2017/09/07/halton-sequences-generating-random-sequences-vii/
// refernce: https://www.shadertoy.com/view/tl2GDw

namespace rnd {

    __host__ __device__ inline uint32_t hash32(uint32_t x) {
        x ^= 61u ^ (x >> 16);
        x *= 9u;
        x ^= x >> 4;
        x *= 0x27d4eb2dU;
        x ^= x >> 15;
        return x;
    }

    __host__ __device__ inline float u01(uint32_t x) { return (x >> 8) * (1.0f / 16777216.0f); }

    __host__ __device__ inline float bitReverse(uint32_t n) {
        n = (n << 16) | (n >> 16);
        n = ((n & 0x00ff00ffu) << 8) | ((n & 0xff00ff00u) >> 8);
        n = ((n & 0x0f0f0f0fu) << 4) | ((n & 0xf0f0f0f0u) >> 4);
        n = ((n & 0x33333333u) << 2) | ((n & 0xccccccccu) >> 2);
        n = ((n & 0x55555555u) << 1) | ((n & 0xaaaaaaaau) >> 1);
        return (float)n * (1.0f / 4294967296.0f);
    }

    __host__ __device__ inline float radicalInverse(uint32_t base, uint32_t n) {
        float invB = 1.0f / (float)base;
        float invP = invB, value = 0.0f;
        while (n) {
            value += (float)(n % base) * invP;
            n /= base;
            invP *= invB;
        }
        return value;
    }

    __host__ __device__ inline uint32_t haltonPrime(uint32_t dim) {
        static const uint16_t p[] = {
            2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
            59,61,67,71,73,79,83,89,97,101,103,107,109,113
        };
        uint32_t n = (uint32_t)(sizeof(p) / sizeof(p[0]));
        return (dim < n) ? p[dim] : 2u;
    }

    __host__ __device__ inline float halton(int dim, int index) {
        uint32_t d = (dim < 0) ? 0u : (uint32_t)dim;
        uint32_t si = (index < 0) ? 0u : (uint32_t)index;
        uint32_t base = haltonPrime(d);
        return (base == 2u) ? bitReverse(si) : radicalInverse(base, si);
    }

    __host__ __device__ inline float fract(float x) { return x - floor(x); }

    __host__ __device__ inline float halton_rng(int dim, int index, int iter, int depth) {
        uint32_t d = (dim < 0) ? 0u : (uint32_t)dim;
        uint32_t si = (index < 0) ? 0u : (uint32_t)index;
        uint32_t it = (iter < 0) ? 0u : (uint32_t)iter;
        uint32_t fs = (depth < 0) ? 0u : (uint32_t)depth;
        float h = halton((int)d, (int)si);
        float shift = u01(hash32(it ^ (d * 0x9E3779B9u) ^ (fs * 0x85ebca6bu)));
        return fract(h + shift);
    }

} 
