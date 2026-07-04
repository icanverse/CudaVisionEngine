#pragma once
#include <cuda_runtime.h>
#include <vector>

struct Particle {
    float3 position;
    float3 velocity;
    float lifetime;      // 1.0 (canlı) -> 0.0 (ölü)
};

class ParticleSystem {
private:
    int numParticles;
    Particle* d_particles; // VRAM'deki parçacık havuzu

public:
    ParticleSystem(int count);
    ~ParticleSystem();

    void update(float deltaTime, float timeTracker);
    void draw(unsigned char* vram, int width, int height);

    // Çizim için veriyi dışarıya (veya OpenGL'e) açan getter
    Particle* getDevicePointer() { return d_particles; }
};

__global__ void updateKernel(Particle* particles, int numParticles, float deltaTime);
__global__ void drawKernel(Particle* particles, int numParticles, unsigned char* vram, int width, int height);

