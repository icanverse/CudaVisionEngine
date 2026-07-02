#include "../../../include/Compute/ParticleSystem/ParticleSystem.cuh"


ParticleSystem::ParticleSystem(int count) : numParticles(count) {
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    // CPU tarafında geçici bir dizi oluştur (Veriyi GPU'ya itmek için köprü)
    std::vector<Particle> h_particles(numParticles);
    for (int i = 0; i < numParticles; i++) {
        // Pozisyon: Merkezden (0,0,0) hafifçe sapmış
        h_particles[i].position = make_float3(0.0f, 0.0f, 0.0f);

        // Hız: Rastgele yönlerde (kıvılcım etkisi için -1 ile 1 arası)
        float rx = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float ry = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float rz = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_particles[i].velocity = make_float3(rx, ry, rz);

        // Ömür farklı zamanda ölsün diye rastgele
        h_particles[i].lifetime = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_particles, h_particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
}

ParticleSystem::~ParticleSystem() {

}

void ParticleSystem::draw(unsigned char* vram, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // drawKernel çekirdeğini buradan ateşliyoruz
    drawKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, vram, width, height);
}

void ParticleSystem::update(float deltaTime) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    updateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime);
}

__global__ void updateKernel(Particle* particles, int numParticles, float deltaTime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numParticles) {
        particles[idx].position.x += particles[idx].velocity.x * deltaTime;
        particles[idx].position.y += particles[idx].velocity.y * deltaTime;
        particles[idx].position.z += particles[idx].velocity.z * deltaTime;

        particles[idx].lifetime -= deltaTime * 0.5f;
    }
}

__global__ void drawKernel(Particle* particles, int numParticles, unsigned char* vram, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles && particles[idx].lifetime > 0.0f) {
        //  Pozisyonu piksel koordinatına dönüştür (Basit bir projeksiyon)
        // Parçacık pozisyonu -1.0 ile 1.0 arasındaysa, bunu ekran çözünürlüğüne haritalıyoruz
        int px = (int)((particles[idx].position.x + 1.0f) * 0.5f * width);
        int py = (int)((particles[idx].position.y + 1.0f) * 0.5f * height);

        //  Sınırları kontrol et
        if (px >= 0 && px < width && py >= 0 && py < height) {
            // 3. Pikselleri Turuncu (RGB: 255, 165, 0) boya
            int pixelOffset = (py * width + px) * 3;
            vram[pixelOffset + 0] = 255; // R
            vram[pixelOffset + 1] = 165; // G
            vram[pixelOffset + 2] = 0;   // B
        }
    }
}