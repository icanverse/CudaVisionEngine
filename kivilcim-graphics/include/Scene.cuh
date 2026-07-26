#ifndef CUDAVISIONENGINE_SCENE_CUH
#define CUDAVISIONENGINE_SCENE_CUH

#pragma once

#include <vector>
#include "Types3D.cuh"

class Scene {
private:
    std::vector<Object3D> objects;
    std::vector<PointLight> lights;
    Camera activeCamera;

public:
    Scene();
    ~Scene();

    Scene& addObject(const float3* cpu_vertices, int numVerticals,
                     const int3* cpu_indices, int numTris,
                     float3 pos, float3 rot, Material mat);

    Scene& addLight(float3 pos, float3 color, float intensity);

    Scene& clear();

    const std::vector<Object3D>& getObjects() const { return objects; }
    const std::vector<PointLight>& getLights() const { return lights; }

    Scene& setCamera(float3 pos, float3 rot);
    Camera getCamera() const { return activeCamera; }
};

#endif //CUDAVISIONENGINE_SCENE_CUH