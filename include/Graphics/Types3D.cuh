#ifndef CUDAVISIONENGINE_TYPES3D_CUH
#define CUDAVISIONENGINE_TYPES3D_CUH
#include <vector_types.h>

struct Camera {
    float3 position;
    float3 rotation;
};

struct PointLight {
    float3 position;
    float3 color;
    float intensity;
};

struct Object3D {
    float3* d_vertices;
    int3* d_indices;
    int numTriangles;

    float3 position;
    float3 rotation;
    float3 ambient_color;
};

#endif //CUDAVISIONENGINE_TYPES3D_CUH