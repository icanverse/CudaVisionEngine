#ifndef CUDAVISIONENGINE_ELEMENTARYNODES_H
#define CUDAVISIONENGINE_ELEMENTARYNODES_H

#include <cuda_runtime.h>
#include <math_functions.h>
#include "Shaders.cuh"
#include "ColorSpaceConverter.cuh"

//
// >> Renk Düğümleri
//

__device__ inline float3 nRGBColor(float r, float g, float b) {
    return make_float3(r, g, b);
}

__device__ inline float3 nHSVColor(float h, float s, float v) {
    return make_float3(h, s, v);
}

//
// >> Gürültü Düğümleri
//

__device__ inline float nWhiteNoise(float3 pos, float scale, float time) {
    float randomHash = pseudoRandomHash3D(pos);

    float px = (pos.x * scale) + time * randomHash;
    float py = (pos.y * scale) + time * randomHash;
    float pz = (pos.z * scale) + time * randomHash;
    float3 p = {px, py, pz};

    return pseudoRandomHash3D(p);
}

__device__ inline void nStaticTV(float &r, float &g, float &b, float3 hitPoint, float time, float noiseScale) {
    float noiseVal = nWhiteNoise(hitPoint, noiseScale, time);

    r *= noiseVal;
    g *= noiseVal;
    b *= noiseVal;
}

__device__ inline void sQuantumGlitch(float &r, float &g, float &b, float3 hitPoint, float time, float noiseScale) {
    float noiseValue = nWhiteNoise(hitPoint, noiseScale, time);

    if (noiseValue > 0.85f) {
        r = 0.2f;
        g = 2.0f;
        b = 2.5f;
    } else if (noiseValue < 0.15f) {
        // Siyah boşluklar (Delikler)
        r *= 0.1f;
        g *= 0.1f;
        b *= 0.1f;
    }
}

//
// >> Gradyan Düğümleri
//

__device__ inline float3 nLerpColor(float3 coloA, float3 coloB, float t) {
    float safe_t = fmaxf(0.0f, fminf(t, 1.0f));

    return make_float3(
            colorA.x + (colorB.x - colorA.x) * safe_t,
            colorA.y + (colorB.y - colorA.y) * safe_t,
            colorA.z + (colorB.z - colorA.z) * safe_t
        );
}

__device__ inline float nFresnel(float3 viewDir, float3 normal, float power) {
    float dotProduct = (viewDir.x * normal.x) + (viewDir.y * normal.y) + (viewDir.z * normal.z);
    float facing = 1.0f - fmaxf(dotProduct, 0.0f);
    return powf(facing, power);
}

__device__ inline float nSmoothstep(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf((x - edge0) / (edge1 - edge0), 1.0f));
    return t * t * (3.0f - 2.0f * t);
}

#endif //CUDAVISIONENGINE_ELEMENTARYNODES_H