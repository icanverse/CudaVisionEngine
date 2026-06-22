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

struct Material {
    float3 color;      // Temel renk ~albedo
    float ambient;     // Ortam Işığı Yansıtma
    float diffuse;     // Matlık
    float specular;    // Parlama Şiddeti
    float shininess;    // Parlaklık Odağı

    int effectType;
    float effectParam1;
    float effectParam2;
};

struct Object3D {
    float3* d_vertices;
    int3* d_indices;
    int numTriangles;

    float3 position;
    float3 rotation;

    // aabb sınırları (performans için)
    float3 aabbMin; // Kutunun minimum x,y,z koordinatları
    float3 aabbMax; // Kutunun maksimum x,y,z koordinatları

    Material material;

};

#endif //CUDAVISIONENGINE_TYPES3D_CUH