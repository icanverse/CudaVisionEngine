#include "Compute/ParticleSystem/ParticleSystem.cuh"

#include <cstdlib>
#include <cmath> // sinf ve cosf için eklendi

__global__ void updateKernel(Particle* particles, int numParticles, float deltaTime, float timeTracker) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles) {
        float phase = (float)idx;

        // Organik süzülme dalgalanmaları
        float driftX = sinf(timeTracker * 2.0f + phase) * 0.5f;
        float driftY = cosf(timeTracker * 1.5f + phase * 0.5f) * 0.5f;
        float driftZ = sinf(timeTracker * 1.8f - phase) * 0.5f;

        // Pozisyon güncellemesi
        particles[idx].position.x += (particles[idx].velocity.x + driftX) * deltaTime;
        particles[idx].position.y += (particles[idx].velocity.y + driftY) * deltaTime;
        particles[idx].position.z += (particles[idx].velocity.z + driftZ) * deltaTime;

        // Ömür azaltma
        particles[idx].lifetime -= deltaTime * 0.2f;

        // Yeniden Doğuş Mekaniği (Merkezden tekrar çıkış)
        if (particles[idx].lifetime <= 0.0f) {
            particles[idx].position = make_float3(0.0f, 0.0f, 0.0f);
            particles[idx].lifetime = 1.0f;
        }
    }
}

// Çizim (Nokta basma) Kernel'ı
__global__ void drawKernel(Particle* particles, int numParticles, unsigned char* vram, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles && particles[idx].lifetime > 0.0f) {
        int px = (int)((particles[idx].position.x + 1.0f) * 0.5f * width);
        int py = (int)((particles[idx].position.y + 1.0f) * 0.5f * height);

        if (px >= 0 && px < width && py >= 0 && py < height) {
            int pixelOffset = (py * width + px) * 3;
            vram[pixelOffset + 0] = 255; // R
            vram[pixelOffset + 1] = 165; // G
            vram[pixelOffset + 2] = 0;   // B
        }
    }
}

ParticleSystem::ParticleSystem(int count) : numParticles(count) {
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    std::vector<Particle> h_particles(numParticles);
    for (int i = 0; i < numParticles; i++) {
        h_particles[i].position = make_float3(0.0f, 0.0f, 0.0f);

        float rx = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float ry = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float rz = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_particles[i].velocity = make_float3(rx, ry, rz);

        h_particles[i].lifetime = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_particles, h_particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
}

ParticleSystem::~ParticleSystem() {
    cudaFree(d_particles);
}

// Fiziği ateşleyen fonksiyon (timeTracker eklendi)
void ParticleSystem::update(float deltaTime, float timeTracker) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // 4 parametreli çağrı yukarıdaki 4 parametreli tanım ile eşleşiyor!
    updateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime, timeTracker);
}

// Ekrana basan köprü fonksiyon
void ParticleSystem::draw(unsigned char* vram, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    drawKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, vram, width, height);
}