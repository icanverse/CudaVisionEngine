#ifndef CUDAVISIONENGINE_RENDERER_CUH
#define CUDAVISIONENGINE_RENDERER_CUH

#pragma once
#include "Scene.cuh"

class Renderer3D {
private:
    int width;
    int height;
    int channels;

public:
    Renderer3D(int w, int h, int c);
    ~Renderer3D();

    void render(float* d_data, const Scene& scene, float time);
};

#endif
