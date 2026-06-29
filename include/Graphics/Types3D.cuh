#ifndef CUDAVISIONENGINE_TYPES3D_CUH
#define CUDAVISIONENGINE_TYPES3D_CUH
#include <vector_types.h>
#include <stdint.h> // uint32_t kullanabilmek için EKLENDİ

struct Camera {
    float3 position;
    float3 rotation;

    bool isOrthographic = false; // true yaparsan 2D UI moduna geçer
    float orthoSize = 10.0f;     // Ekrana ne kadar alan sığacağı (Zoom)
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
    float shininess;   // Parlaklık Odağı

    // --- YENİ BİTMASK SİSTEMİ ---
    uint32_t effectFlags;

    // Efekt Parametreleri Havuzu
    float glowSpeed;
    float scanFreq; float scanSpeed;
    float tronGridSize; float tronThickness;
    float radarFreq; float radarSpeed;
    float jitterIntensity;
    float dissolveSpeed;
    float liquidFreq; float liquidSpeed;

    // YENİ EFEKT PARAMETRELERİ
    float celBands;
    float fogStart; float fogEnd; float fogDensity; float3 fogColor;
    float3 shieldColor; float rimPower; float rimIntensity;
    float noiseScale;

};

struct Object3D {
    float3* d_vertices;
    int3* d_indices;
    int numTriangles;

    float3 position;
    float3 rotation;

    float3 aabbMin;
    float3 aabbMax;

    Material material;
};

#endif //CUDAVISIONENGINE_TYPES3D_CUH