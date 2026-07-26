#include "../include/Renderer3D.cuh"
#include <iostream>

__global__ void renderMultiObjectMesh(float* d_data, int width, int height, int channels,
                                      Object3D* d_objects, int numObjects,
                                      PointLight* d_lights, int numLights,
                                      Camera cam, float time);

Renderer3D::Renderer3D(int w, int h, int c) : width(w), height(h), channels(c) {
    std::cout << "[Renderer3D] Grafik Motor Katmani Baslatildi (" << w << "x" << h << ")" << std::endl;
}

Renderer3D::~Renderer3D() {
}

void Renderer3D::render(float* d_data, const Scene& scene, float time) {
    int numObjects = scene.getObjects().size();
    int numLights = scene.getLights().size();

    // Sahnede obje yoksa çizilmez
    if (numObjects == 0) return;

    // listeleri (vector) ham dizilere (Array)'e çevireceğiz :: CPU -> GPU
    Object3D* d_objects;
    PointLight* d_lights;

    cudaMalloc((void**)&d_objects, numObjects * sizeof(Object3D));
    cudaMemcpy(d_objects, scene.getObjects().data(), numObjects * sizeof(Object3D), cudaMemcpyHostToDevice);

    if (numLights > 0) {
        cudaMalloc((void**)&d_lights, numLights * sizeof(PointLight));
        cudaMemcpy(d_lights, scene.getLights().data(), numLights * sizeof(PointLight), cudaMemcpyHostToDevice);
    } else {
        d_lights = nullptr;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    cudaMemset(d_data, 0, width * height * channels * sizeof(float));

    renderMultiObjectMesh<<<gridSize, blockSize>>>(d_data, width, height, channels,
                                                    d_objects, numObjects,
                                                    d_lights, numLights,
                                                    scene.getCamera(), time);

    cudaDeviceSynchronize();

    cudaFree(d_objects);
    if (numLights > 0) cudaFree(d_lights);
}