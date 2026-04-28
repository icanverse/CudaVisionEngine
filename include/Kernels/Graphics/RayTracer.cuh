#ifndef CUDAVISIONENGINE_RAYTRACER_CUH
#define CUDAVISIONENGINE_RAYTRACER_CUH

#pragma once
#include <cuda_runtime.h>

__device__ inline float3 crossProduct(float3 a, float3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ inline float dotProduct(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 sub(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

// --- Möller-Trumbore Kesişim Algoritması ---
__device__ inline bool intersectTriangle(float3 rayO, float3 rayD, float3 v0, float3 v1, float3 v2, float &t) {
    const float EPSILON = 0.0000001f;
    float3 edge1 = sub(v1, v0);
    float3 edge2 = sub(v2, v0);
    float3 h = crossProduct(rayD, edge2);
    float a = dotProduct(edge1, h);
    if (a > -EPSILON && a < EPSILON) return false;
    float f = 1.0f / a;
    float3 s = sub(rayO, v0);
    float u = f * dotProduct(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = crossProduct(s, edge1);
    float v = f * dotProduct(rayD, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * dotProduct(edge2, q);
    return (t > EPSILON);
}

__global__ void renderMesh(float* d_data, int width, int height, int channels,
                           float3* d_vertices, int3* d_indices, int numTriangles, float time);

#endif //CUDAVISIONENGINE_RAYTRACER_CUH