#ifndef CUDAVISIONENGINE_RAYTRACER_CUH
#define CUDAVISIONENGINE_RAYTRACER_CUH

#pragma once
#include <cuda_runtime.h>

__device__ inline float3 crossProduct(float3 a, float3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ inline float dotProduct(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 sub(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

// --- Möller-Trumbore Kesişim Algoritması (Epsilon Toleranslı) ---
__device__ inline bool intersectTriangle(float3 rayO, float3 rayD, float3 v0, float3 v1, float3 v2, float &t) {
    const float EPSILON = 0.0000001f;
    const float BARY_EPSILON = 0.001f; // YENİ: Kıymık üçgenler için tolerans payı!

    float3 edge1 = sub(v1, v0);
    float3 edge2 = sub(v2, v0);
    float3 h = crossProduct(rayD, edge2);
    float a = dotProduct(edge1, h);

    // Işın üçgene paralel mi?
    if (a > -EPSILON && a < EPSILON) return false;

    float f = 1.0f / a;
    float3 s = sub(rayO, v0);

    // Kütle Merkezi 1 (U) -> Sıfır tolerans yerine BARY_EPSILON esnekliği
    float u = f * dotProduct(s, h);
    if (u < -BARY_EPSILON || u > 1.0f + BARY_EPSILON) return false;

    float3 q = crossProduct(s, edge1);

    // Kütle Merkezi 2 (V) -> Sıfır tolerans yerine BARY_EPSILON esnekliği
    float v = f * dotProduct(rayD, q);
    if (v < -BARY_EPSILON || u + v > 1.0f + BARY_EPSILON) return false;

    t = f * dotProduct(edge2, q);
    return (t > EPSILON);
}

__global__ void renderMesh(float* d_data, int width, int height, int channels,
                           float3* d_vertices, int3* d_indices, int numTriangles, float time);

#endif //CUDAVISIONENGINE_RAYTRACER_CUH