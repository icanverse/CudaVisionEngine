#include "Compute/ParticleSystem/ParticleSystem.cuh"
// Yeni yazdığın kernellerin prototiplerini içeren başlık dosyasını buraya eklemelisin
#include "Compute/ParticleSystem/ParticleSystem.cuh"
#include <cstdlib>
#include <cmath>
#include <vector>

ParticleSystem::ParticleSystem(int count) : numParticles(count) {
    cudaMalloc(&d_particles, numParticles * sizeof(Particle));

    std::vector<Particle> h_particles(numParticles);
    for (int i = 0; i < numParticles; i++) {

        // Pozisyon Atamaları
        float px = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float py = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float pz = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_particles[i].position = make_float3(px, py, pz);
        h_particles[i].initialPosition = make_float3(px, py, pz); // Doğduğu yeri kaydet

        // Hız ve İvme (Başlangıçta ivme sıfırdır)
        float rx = ((float)rand() / RAND_MAX) * 1.0f - 1.0f;
        float ry = ((float)rand() / RAND_MAX) * 1.0f - 1.0f;
        float rz = ((float)rand() / RAND_MAX) * 0.4f - 1.0f;
        h_particles[i].velocity = make_float3(rx * 0.5f, ry * 0.5f, rz * 0.5f);
        h_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);

        // Boyut ve Parlaklık (Min: 2.0, Max: 6.0 olsun)
        h_particles[i].size = make_float3(30.0f, 28.0f, 36.0f);
        h_particles[i].intensity = 1.0f;

        // Renk Ataması (Ateş Böceği için Rastgele Sarımsı/Turuncu Tonlar)
        // Kırmızı sabit 255, Yeşil rastgele (100-200 arası), Mavi 0
        float randomGreen = 100.0f + ((float)rand() / RAND_MAX) * 100.0f;
        h_particles[i].color = make_float3(255.0f, randomGreen, 0.0f);

        h_particles[i].lifetime = (float)rand() / RAND_MAX;
    }

    // CPU'daki (Host) veriyi GPU'ya (Device) kopyala
    cudaMemcpy(d_particles, h_particles.data(), numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
}

ParticleSystem::~ParticleSystem() {
    cudaFree(d_particles);
}

void ParticleSystem::update(float deltaTime, float timeTracker) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Modüler Mimari Devrede: Önce Davranış, Sonra Fizik!

    // Davranış Katmanı (Rüzgar Gücü 0.5f, Dim ve Pulse aktif)
    pBehaviorLayer<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, timeTracker, 0.24f, true, true);

    cudaDeviceSynchronize();

    // Kinematik Katmanı
    pKinematicLayer<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime, timeTracker, true, true);
}

void ParticleSystem::draw(unsigned char* vram, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // 3. Çizim Katmanı (Güncellenmiş pRenderLayer)
    pRenderLayer<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, vram, width, height, 0, 0);
}