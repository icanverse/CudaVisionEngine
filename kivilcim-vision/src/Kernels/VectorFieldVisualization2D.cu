#include "../include/Kernels/VectorFieldVisualization2D.cuh"

// Heat Map
__global__ void applyVectorFieldColoring(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity) {

    int dx = threadIdx.x + blockDim.x * blockIdx.x;
    int dy = threadIdx.y + blockDim.y * blockIdx.y;

    int margin = 5;
    if (dx < margin || dy < margin || dx >= width - margin || dy >= height - margin) return;

    int index1D = dy * width + dx;
    int index3D = index1D * channels;

    float u = flowU[index1D];
    float v = flowV[index1D];

    float magnitude = sqrtf((u * u) + (v * v)) * intensity;

    if (magnitude < 0.2f) return;

    magnitude = min(1.0f, magnitude);

    float angle = atan2f(v, u);
    if (angle < 0.0f) {
        angle += 2.0f * PI;
    }

    float hue = angle / (2.0f * PI);

    float r = 0.0f, g = 0.0f, b = 0.0f;
    float h_val = hue * 6.0f; // Renk çemberini 6 dilime (sektöre) böl
    int i = (int)h_val;       // Hangi dilimdeyiz? (0, 1, 2, 3, 4, 5)
    float f = h_val - i;      // Dilim içindeki küsurat
    float q = 1.0f - f;       // Ters küsurat

    // Bulunduğumuz sektöre göre RGB ağırlıklarını dağıt
    switch (i % 6) {
        case 0: r = 1.0f; g = f;    b = 0.0f; break; // Kırmızıdan Sarıya
        case 1: r = q;    g = 1.0f; b = 0.0f; break; // Sarıdan Yeşile
        case 2: r = 0.0f; g = 1.0f; b = f;    break; // Yeşilden Cyana
        case 3: r = 0.0f; g = q;    b = 1.0f; break; // Cyandan Maviye
        case 4: r = f;    g = 0.0f; b = 1.0f; break; // Maviden Magentaya
        case 5: r = 1.0f; g = 0.0f; b = q;    break; // Magentadan Kırmızıya
    }

    // RGB renklerini, hareketin şiddetiyle (magnitude) çarpıp mevcut resmin üzerine ekliyoruz.
    d_data[index3D]     = min(1.0f, d_data[index3D]     + (r * magnitude));
    d_data[index3D + 1] = min(1.0f, d_data[index3D + 1] + (g * magnitude));
    d_data[index3D + 2] = min(1.0f, d_data[index3D + 2] + (b * magnitude));
}

// Yüzey Normali
__global__ void applyNormalMapVisualization(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;

    int margin = 5;
    if (dx < margin || dy < margin || dx >= width - margin || dy >= height - margin) return;

    int index1D = dy * width + dx;
    int index3D = index1D * channels;

    float u = flowU[index1D] * intensity;
    float v = flowV[index1D] * intensity;

    float normal_x = max(min(1.0f, u), -1.0f);
    float normal_y = max(min(1.0f, v), -1.0f);

    float normal_zz = 1.0f - normal_x * normal_x - normal_y * normal_y;
    float normal_z;

    if (normal_zz <= 0.0f) {
        normal_z = 0.0f;
    } else {
        normal_z = sqrtf(normal_zz);
    }

    normal_x = ( normal_x + 1.0f ) / 2.0f;
    normal_y = ( normal_y + 1.0f ) / 2.0f;
    normal_z = ( normal_z + 1.0f ) / 2.0f;

    float r = normal_x;
    float g = normal_y;
    float b = normal_z;

    d_data[index3D]     = r;
    d_data[index3D + 1] = g;
    d_data[index3D + 2] = b;

}

// Seyrek Vektör Okları
__global__ void applyQuiverPlotVisualization(float* data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity) {
    int dx = threadIdx.x + blockDim.x * blockIdx.x;
    int dy = threadIdx.y + blockDim.y * blockIdx.y;

    int margin = 4;
    if (dx < margin || dy < margin || dx >= width - margin || dy >= height - margin) return;

    int center_x = (dx / 16) * 16 + 8;
    int center_y = (dy / 16) * 16 + 8;

    int index1D = dy * width + dx;
    int index3D = index1D * channels;

    int centerIndex = center_y * width + center_x;

    float u = flowU[centerIndex] * intensity;
    float v = flowV[centerIndex] * intensity;

    float end_x = center_x + u;
    float end_y = center_y + v;

    /// Çizgi Oluşturma Bloğu

    // Çizgi Başlangıcı
    float ax = center_x;
    float ay = center_y;

    // Çizgi Bitişi
    float bx = end_x;
    float by = end_y;

    // Pikselin çizgiye olan vektörü
    float px = dx - ax;  float py = dy - ay;

    // Çizginin kendi vektörü
    float lx = bx - ax;  float ly = by - ay;

    // Çizginin uzunluğunun karesi
    float len_sq = lx * lx + ly * ly;
    if (len_sq < 0.1f) return;

    // Pikselin çizgi üzerindeki izdüşüm oranı (0.0 = Başlangıç, 1.0 = Bitiş)
    float t = max(0.0f, min(1.0f, (px * lx + py * ly) / len_sq));

    // Çizgi üzerindeki en yakın nokta
    float closest_x = ax + t * lx;
    float closest_y = ay + t * ly;

    float dist = sqrtf((dx - closest_x) * (dx - closest_x) + (dy - closest_y) * (dy - closest_y));

    if (dist < 1.0f) {
        data[index3D]     = 1.0f;
        data[index3D + 1] = 0.0f;
        data[index3D + 2] = 0.0f;
    }
}

__global__ void applyJetScalarColorPalette(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float maxSpeed) {
    int dx = threadIdx.x + blockDim.x * blockIdx.x;
    int dy = threadIdx.y + blockDim.y * blockIdx.y;
    int margin = 5;

    if (dx < margin || dy < margin || dx >= width - margin || dy >= height - margin) return;

    int index1D = dy * width + dx;
    int index3D = index1D * channels;

    float u = flowU[index1D];
    float v = flowV[index1D];
    float magnitude = sqrtf(u * u + v * v);

    float val = fminf(1.0f, magnitude / maxSpeed);

    float r = fminf(1.0f, fmaxf(0.0f, 1.5f - fabsf(4.0f * val - 3.0f)));
    float g = fminf(1.0f, fmaxf(0.0f, 1.5f - fabsf(4.0f * val - 2.0f)));
    float b = fminf(1.0f, fmaxf(0.0f, 1.5f - fabsf(4.0f * val - 1.0f)));

    d_data[index3D]     = r;
    d_data[index3D + 1] = g;
    d_data[index3D + 2] = b;
}

__device__ float random_noise(int x, int y) {
    float res = sinf(x * 12.9898f + y * 78.233f) * 43758.5453123f;
    return res - floorf(res);
}

__global__ void applyLineIntegralConvolution(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, int steps) {
    int dx = threadIdx.x + blockDim.x * blockIdx.x;
    int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx >= width || dy >= height) return;

    float total_noise = 0.0f;
    float current_x = (float)dx;
    float current_y = (float)dy;

    for (int i = 0; i < steps; i++) {
        // anlık koordinattaki hız
        int ix = (int)current_x;
        int iy = (int)current_y;

        if (ix < 0 || ix >= width || iy < 0 || iy >= height) break;

        int internal_index = ix + iy * width;
        float u = flowU[internal_index];
        float v = flowV[internal_index];

        total_noise += u * random_noise(ix, iy);

        current_x += u * 0.5f;
        current_y += v * 0.5f;
    }

    float final_lum = total_noise / float(steps);

    int index3D = (dy * width + dx) * channels;

    d_data[index3D]     *= final_lum * 2.0f;
    d_data[index3D + 1] *= final_lum * 2.0f;
    d_data[index3D + 2] *= final_lum * 2.0f;


}