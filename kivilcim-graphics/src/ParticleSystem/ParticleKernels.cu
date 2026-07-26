#include "ParticleSystem/ParticleSystem.cuh"

//
// >>> Davranış Katmanı
//

__device__ inline void pseudoNoise(int idx, float timeTracker, float windForce, float3& inWind) {
    float phase = float(idx);
    float windX = sinf(timeTracker * 2.0f + phase) * windForce;
    float windY = cosf(timeTracker * 1.5f + (phase * 0.5f)) * windForce;
    float windZ = sinf(timeTracker * 2.0f + (phase * 0.5f)) * windForce;
    inWind.x = windX;
    inWind.y = windY;
    inWind.z = windZ;
}

__device__ inline void dynamicDimension(int idx, float timeTracker, float3 size, float& resultSize) {
    float phase = float(idx);
    float wave = sinf(timeTracker * 2.0f + phase);
    wave = (wave + 1) * 0.5f;
    wave = size.y + ( size.z - size.y) * wave;
    resultSize = wave;
}

__device__ inline void dynamicPulse(int idx, float timeTracker, float& pulse ){
    float phase = float(idx);
    float wavePulse = sinf(timeTracker * 4.0f + phase);
    wavePulse = (wavePulse + 1) * 0.5f;
    pulse = wavePulse;
}

__global__ void pBehaviorLayer(Particle* particles, int numParticles,
                               float timeTracker, float windForce,
                               bool isDynamicDim, bool isPulse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles && particles[idx].lifetime > 0.0f) {
        float3 currentWind;

        pseudoNoise(idx, timeTracker, windForce, currentWind);

        particles[idx].acceleration.x += currentWind.x;
        particles[idx].acceleration.y += currentWind.y;
        particles[idx].acceleration.z += currentWind.z;

        if (isDynamicDim == true) {
            float currentSize = particles[idx].size.x;
            float resultSize;
            dynamicDimension(idx, timeTracker,  particles[idx].size, resultSize);
            particles[idx].size.x = resultSize;
        }

        if (isPulse == true) {
            particles[idx].intensity = 0.2f;
            float pulse;
            dynamicPulse(idx, timeTracker, pulse);
            particles[idx].intensity += pulse * 0.8f;
        }
    }
}

//
// >>> Kinematik Katmanı
//

__device__ inline void calculateVelocity(int idx, float deltaTime,
                                         float3& velocity, float3& acceleration) {
    velocity.x = velocity.x + acceleration.x * deltaTime;
    velocity.y = velocity.y + acceleration.y * deltaTime;
    velocity.z = velocity.z + acceleration.z * deltaTime;
}


__device__ inline void calculatePosition(int idx, float deltaTime,
                                                   float3& position, float3 velocity) {
    position.x = position.x + velocity.x * deltaTime;
    position.y = position.y + velocity.y * deltaTime;
    position.z = position.z + velocity.z * deltaTime;
}

__device__ inline float pseudoRandomValue(float seed) {
    float s = sinf(seed * 12.9898f) * 43758.5453f;
    return (s - floorf(s)) * 2.0f - 1.0f; // -1.0 ile 1.0 arasında rastgele değer döndürür
}

__global__ void pKinematicLayer(Particle* particles, int numParticles,
                                float deltaTime, float timeTracker,
                                bool isInfinite, bool isDeadParticleRand) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 zero3 = {0.0f, 0.0f, 0.0f};

    if (idx < numParticles && particles[idx].lifetime > 0.0f) {
        calculateVelocity(idx, deltaTime, particles[idx].velocity, particles[idx].acceleration);
        calculatePosition(idx, deltaTime, particles[idx].position, particles[idx].velocity);
        particles[idx].acceleration = zero3;

        if (isInfinite == false) {
            particles[idx].lifetime -= deltaTime;
        }

        // Ekran dışına çıkan parçacıklar için
        if (particles[idx].position.x > 1.0f || particles[idx].position.x < -1.0f ||
            particles[idx].position.y > 1.0f || particles[idx].position.y < -1.0f) {
            particles[idx].lifetime = -1.0f;
        }
    }
    else if (idx < numParticles) {
        particles[idx].lifetime = 1.0f;
        particles[idx].velocity = zero3;
        particles[idx].acceleration = zero3;

        if (isDeadParticleRand == true) {
            float baseSeed = (float)idx + timeTracker * 1000.0f;
            particles[idx].position.x = pseudoRandomValue(baseSeed);
            particles[idx].position.y = pseudoRandomValue(baseSeed + 1.0f);
            particles[idx].position.z = pseudoRandomValue(baseSeed + 2.0f);
        }
        else {
            particles[idx].position = particles[idx].initialPosition;
        }
    }
}

//
// >>> Render Katmanı
//

__device__ inline int2 convertPositionToPixelCoordinate(float3 position, int width, int height) {
    float nX = (position.x + 1.0f) * 0.5f * (float)width;
    float nY = (position.y + 1.0f) * 0.5f * (float)height;
    return make_int2((int)nX, (int)nY);
}

//
// >>>> Blend (Karıştırma) Modları
//

__device__ inline void blendAdditive(unsigned char* vram, int px, int py, int width,
                                     float3 color, float alpha) {
    int offset = (py * width + px) * 3;

    vram[offset + 0] = (unsigned char)fminf(255.0f, vram[offset + 0] + (color.x * alpha));
    vram[offset + 1] = (unsigned char)fminf(255.0f, vram[offset + 1] + (color.y * alpha));
    vram[offset + 2] = (unsigned char)fminf(255.0f, vram[offset + 2] + (color.z * alpha));
}

__device__ inline void selectBlendMode(int blendMode, unsigned char* vram, int px, int py, int width,
                                       float3 color, float alpha) {
    switch (blendMode) {
        case 0:
            blendAdditive(vram, px, py, width, color, alpha);
            break;
    }
}

//
// >>>> Render (Çizim) Modları
//

__device__ inline void solidBox(Particle* particles, int numParticles,
                                unsigned char* vram, int idx, int2 coordinate,
                                int width, int height, int blendMode) {
    int radius = (int)ceilf(particles[idx].size.x / 2.0f);

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {

            int px = coordinate.x + x;
            int py = coordinate.y + y;

            if (px >= 0 && px < width && py >= 0 && py < height) {
                float activeVisibility = particles[idx].intensity;

                selectBlendMode(blendMode, vram, px, py, width, particles[idx].color, activeVisibility);
            }
        }
    }
}

__device__ inline void glowedCircle(Particle* particles, int numParticles,
                                    unsigned char* vram, int idx, int2 coordinate,
                                    int width, int height, int blendMode) {
    int radius = (int)ceilf(particles[idx].size.x / 2.0f);

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {

            int px = coordinate.x + x;
            int py = coordinate.y + y;

            if (px >= 0 && px < width && py >= 0 && py < height) {

                float dist = sqrtf((float)(x * x + y * y));

                if (dist <= radius) {
                    float baseAlpha = 1.0f - (dist / (float)radius);
                    float activeVisibility = baseAlpha * particles[idx].intensity;

                    selectBlendMode(blendMode, vram, px, py, width, particles[idx].color, activeVisibility);
                }
            }
        }
    }
}

//
// >>>> Ana Render Kerneli
//

__global__ void pRenderLayer(Particle* particles, int numParticles,
                             unsigned char* vram, int width, int height,
                             int blendMode, int renderMode) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles && particles[idx].lifetime > 0.0f) {
        int2 coordinate = convertPositionToPixelCoordinate(particles[idx].position, width, height);

        switch (renderMode) {
            case 0:
                glowedCircle(particles, numParticles, vram, idx, coordinate, width, height, blendMode);
                break;
            case 1:
            default:
                solidBox(particles, numParticles, vram, idx, coordinate, width, height, blendMode);
                break;
        }
    }
}
