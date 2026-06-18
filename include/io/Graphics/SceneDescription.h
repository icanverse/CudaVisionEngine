#pragma once
#include <string>
#include <vector>
#include <set>
#include <vector_types.h>

// Harici dosyaların alışveriş listesi
struct SceneManifest {
    std::set<std::string> meshes;
    std::set<std::string> materials;
    std::set<std::string> cameraFilters;
};

// Tek bir objenin sahnedeki durumunu tutan plan
struct ObjectBlueprint {
    std::string meshPath;
    std::string matPath;
    float3 position;
    float3 rotation;
};

// Işık kaynağı planı
struct LightBlueprint {
    float3 position;
    float3 color;
    float intensity;
};

// Tüm sahne
struct SceneDescription {
    float3 camPos = {0.0f, 0.0f, 0.0f};
    float3 camRot = {0.0f, 0.0f, 0.0f};

    std::vector<LightBlueprint> lights;
    std::vector<ObjectBlueprint> objects;

    SceneManifest manifest;
};