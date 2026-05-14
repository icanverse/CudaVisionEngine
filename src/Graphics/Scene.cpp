#include "../../include/Graphics/Scene.cuh"

#include <cuda_runtime_api.h>
#include <iostream>

Scene::~Scene() {
    clear();
}

Scene& Scene::clear() {
    for (auto& obj : objects) {
        if (obj.d_vertices) cudaFree(obj.d_vertices);
        if (obj.d_indices) cudaFree(obj.d_indices);
    }
    objects.clear();
    lights.clear();
    return *this;
}

Scene& Scene::addObject(const float3* cpu_vertices, int numVerts, 
                        const int3* cpu_indices, int numTris,
                        float3 pos, float3 rot, Material mat) {
    Object3D obj;
    obj.numTriangles = numTris;
    obj.position = pos;
    obj.rotation = rot;
    obj.material = mat;

    cudaMalloc((void**)&obj.d_vertices, numVerts * sizeof(float3));
    cudaMemcpy(obj.d_vertices, cpu_vertices, numVerts * sizeof(float3), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&obj.d_indices, numTris * sizeof(int3));
    cudaMemcpy(obj.d_indices, cpu_indices, numTris * sizeof(int3), cudaMemcpyHostToDevice);

    objects.push_back(obj);
    return *this;
}

Scene& Scene::addLight(float3 pos, float3 color, float intensity) {
    PointLight light = {pos, color, intensity};
    lights.push_back(light);
    return *this;
}

Scene::Scene() {
    // Motor başladığında kamera merkezde dursun
    activeCamera.position = {0.0f, 0.0f, 0.0f};
    activeCamera.rotation = {0.0f, 0.0f, 0.0f};
}

Scene& Scene::setCamera(float3 pos, float3 rot) {
    activeCamera.position = pos;
    activeCamera.rotation = rot;
    return *this;
}