#ifndef CUDAVISIONENGINE_SHADERS_CUH
#define CUDAVISIONENGINE_SHADERS_CUH

#pragma once
#include <cuda_runtime.h>
#include <math_functions.h>

// ==========================================
// KIVILCIM SHADER LABORATUVARI
// ==========================================

__device__ inline void sGlow(float& r, float& g, float& b, float time, float speed) {
    // Nefes alan yumuşak bir parlama (Pulse)
    float pulse = (sinf(time * speed) + 1.0f) * 0.5f;
    float emissive = pulse * 0.4f; // Kendi içinden parlama şiddeti

    r += emissive;
    g += emissive;
    b += emissive;
}

__device__ inline void sScanlines(float& r, float& g, float& b, float yPos, float time, float freq, float speed) {
    // Yüksekliğe ve zamana bağlı akan siber-enerji halkaları
    float wave = (sinf(yPos * freq + time * speed) + 1.0f) * 0.5f;
    wave = powf(wave, 8.0f); // Çizgileri jilet gibi keskinleştir

    r += wave * 0.8f;
    g += wave * 0.8f;
    b += wave * 0.8f;
}

__device__ inline void sNegativeZone(float& r, float& g, float& b) {
    r = 1 - r;
    g = 1 - g;
    b = 1 - b;
}

__device__ inline void sRGBDisco(float & r, float& g, float& b, float time) {
    float r_wave = (sinf(time) + 1.0f) * 0.5f;
    float g_wave = (sinf(time * 2.0f) + 1.0f) * 0.5f;
    float b_wave = (sinf(time * 4.0f) + 1.0f) * 0.5f;

    r = r * r_wave;
    g = g * g_wave;
    b = b * b_wave;
}

// nx, ny, nz normaller
__device__ inline void sNormalDebugger(float& r, float& g, float& b, float nx, float ny, float nz) {
    r = (nx + 1.0f) * 0.5f;
    g = (ny + 1.0f) * 0.5f;
    b = (nz + 1.0f) * 0.5f;
}

__device__ inline void sCelShading(float& r, float& g, float& b, float light_intensity, float bands) {
    light_intensity = fmaxf(0.0f, fminf(1.0f, light_intensity));
    float stepped_light = floorf(light_intensity * bands) / bands;
    stepped_light = fmaxf(0.2f, stepped_light);

    r *= stepped_light;
    g *= stepped_light;
    b *= stepped_light;

}

// th1 > th2 > th3 KONTROLÜ UNUTULMASIN
__device__ inline void sCelShading_withThreshold(float& r, float& g, float& b, float light_intensity, float3 threshold_band, float4 threshold_mul) {
    float intensity;

    if (light_intensity > threshold_band.x) {
        intensity = threshold_mul.x;
    } else if (light_intensity > threshold_band.y) {
        intensity = threshold_mul.y;
    } else if (light_intensity > threshold_band.z) {
        intensity = threshold_mul.z;
    } else {
        intensity = threshold_mul.w;
    }

    r *= intensity;
    g *= intensity;
    b *= intensity;
}

__device__ inline void sLinearDepthFog(float& r, float& g, float& b, float depth, float fog_start, float fog_end, float3 fog_color) {
    float fog_factor = (depth - fog_start) / (fog_end - fog_start);
    fog_factor = fminf(1.0f,fmaxf(0.0f, fog_factor));

    r = (1.0 - fog_factor) * r + (fog_factor) * fog_color.x;
    g = (1.0 - fog_factor) * r + (fog_factor) * fog_color.y;
    b = (1.0 - fog_factor) * r + (fog_factor) * fog_color.z;
}

__device__ inline void sExponentialDepthFog(float& r, float& g, float& b, float depth, float fog_density, float3 fog_color, int d) {
    float dd = depth * fog_density;
    float power_val = dd;

    if (d == 2) {
        power_val = dd * dd;
    } else if (d > 2) {
        power_val = powf(dd, (float)d);
    }

    float fog_factor = 1.0f - expf(-power_val);
    fog_factor = fmaxf(0.0f, fminf(1.0f, fog_factor));

    r = (1.0f - fog_factor) * r + (fog_factor) * fog_color.x;
    g = (1.0f - fog_factor) * g + (fog_factor) * fog_color.y;
    b = (1.0f - fog_factor) * b + (fog_factor) * fog_color.z;
}

__device__ inline void sFresnelShield(float& r, float& g, float& b,
                                      float nx, float ny, float nz,
                                      float vx, float vy, float vz,
                                      float3 shield_color,
                                      float rim_power,
                                      float rim_intensity) {

    float v_dot_n = (nx * vx) + (ny * vy) + (nz * vz);
    v_dot_n = fmaxf(0.0f, fminf(1.0f, v_dot_n));

    float rim = 1 - v_dot_n;
    float sharp_rim = powf(rim, rim_power);

    r += sharp_rim * shield_color.x * rim_intensity;
    g += sharp_rim * shield_color.y * rim_intensity;
    b += sharp_rim * shield_color.z * rim_intensity;

}

__device__ inline void sTronGrid(float& r, float& g, float& b, float3 hitPoint, float gridSize, float thickness) {
    // Izgara negatif uzayda bozulmasın
    float absX = fabsf(hitPoint.x);
    float absY = fabsf(hitPoint.y);
    float absZ = fabsf(hitPoint.z);

    // Uzay koordinatları modu
    float modX = fmodf(absX, gridSize);
    float modY = fmodf(absY, gridSize);
    float modZ = fmodf(absZ, gridSize);

    //
    if (modX < thickness || modY < thickness || modZ < thickness) {
        r = 0.0f;
        g = 1.0f;
        b = 0.5f;
    }
}

__device__ inline void sRadarPing(float& r, float& g, float& b, float3 hitPoint, float3 objPos, float time, float freq, float speed) {
    // Işığın çaptığı noktanın, nesne merkezine uzaklık farkı
    float dx = hitPoint.x - objPos.x;
    float dy = hitPoint.y - objPos.y;
    float dz = hitPoint.z - objPos.z;

    float dist = sqrtf((dx * dx) + (dy * dy) + (dz * dz));

    float wave = sinf(dist * freq - time * speed);
    wave = (wave + 1.0f) * 0.5f;

    wave = powf(wave, 20.0f);

    r += wave * 0.2f;
    g += wave * 1.0f;
    b += wave * 0.2f;
}

__device__ inline float pseudoRandomHash(float val) {
    float s = sinf(val * 12.9898f) * 43758.5453f;
    return s - floorf(s);
}

__device__ inline void sMatrixJitter(float& r, float& g, float& b, float3 hitPoint, float time, float intensity) {
    // Objenin yüksekliğini şeritlere bölüyoruz.
    // 10.0f şeritlerin kalınlığını belirler.
    float band = floorf(hitPoint.y * 10.0f);

    float noise = pseudoRandomHash(band + time * intensity);

    if (noise > 0.95f) {
        r = 1.0f;
        g = 0.0f;
        b = pseudoRandomHash(time * hitPoint.x);
    }
}

__device__ inline float pseudoRandomHash3D(float3 p) {
    float s = sinf(p.x * 12.9898f + p.y * 78.233f + p.z * 37.719f) * 43758.5453f;
    return s - floorf(s);
}

__device__ inline void sThanosSnapDissolve(float& r, float& g, float& b, float3 hitPoint, float time, float speed) {
    float3 noiseScale = {hitPoint.x * 3.0f, hitPoint.y * 3.0f, hitPoint.z * 3.0f};

    // Bu pikselin uzaysal gürültüsünü alıyoruz.
    float noise = pseudoRandomHash3D(noiseScale);

    // Zamanla 0.0'dan 1.2'ye kadar çıkan ve başa saran bir "Erime Seviyesi"
    float dissolveProgress = fmodf(time * speed, 1.2f);

    // Eğer doku değeri erime seviyesinin altında kaldıysa, o piksel ölmüştür.
    if (noise < dissolveProgress) {
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
    }
    // Burn Edge Piksel henüz ölmemiş ama ölüme çok yakınsa (0.05f tolerans)
    else if (noise < dissolveProgress + 0.05f) {
        r = 1.0f;
        g = 0.4f;
        b = 0.0f;
    }
}

__device__ inline void sLiquidFlow(float& r, float& g, float& b, float3 hitPoint, float time, float flowSpeed, float freq) {
    float mov = time * flowSpeed;

    float f_x = sinf(hitPoint.x * freq + mov * 0.8f);
    float f_y = sinf(hitPoint.y * freq + mov * 0.4f);
    float f_z = sinf(hitPoint.z * freq + mov * 0.5f);

    float sum = f_x + f_y + f_z;
    sum = (sum + 3) / 6;

    sum = powf(sum, 3.0f);

    r += sum * 1.0f;
    g += sum * 0.4f;
    b += sum * 0.1f;
}

__device__ inline void s3DRetroVoxel(float &r, float &g, float &b, float3 hitPoint, float gridSize) {
    float vX = floorf(hitPoint.x * gridSize) / gridSize;
    float vY = floorf(hitPoint.y * gridSize) / gridSize;
    float vZ = floorf(hitPoint.z * gridSize) / gridSize;
    float3 v = {vX, vY, vZ};

    float noiseVal = pseudoRandomHash3D(v);
    noiseVal = noiseVal * 0.3f + 0.6f;

    r += noiseVal;
    g += noiseVal;
    b += noiseVal;

}

__device__ inline void sLidarScanner(float &r, float &g, float &b, float3 hitPoint, float3 sensorPos, float time) {

    float dx = hitPoint.x - sensorPos.x;
    float dy = hitPoint.y - sensorPos.y;
    float dz = hitPoint.z - sensorPos.z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    float freq = 5.0f;
    float speed = 10.0f;

    float wave = sinf(distance * freq - time * speed);
    wave = (wave + 1.0f) * 0.5f;

    float sharpness = 30.0f;
    wave = powf(wave, sharpness);

    r *= 0.05f;
    g *= 0.1f;
    b *= 0.1f;

    r += wave * 0.1f;
    g += wave * 2.5f;
    b += wave * 0.8f;
}


#endif //CUDAVISIONENGINE_SHADERS_CUH