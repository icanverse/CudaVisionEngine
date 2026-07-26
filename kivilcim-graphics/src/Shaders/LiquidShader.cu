#include <device_launch_parameters.h>
#include <math_functions.h>

namespace Kivilcim {
    namespace Shaders {

        __device__ float length2(float x, float y) {
            return sqrtf(x * x + y * y);
        }

        __device__ float mix(float a, float b, float t) {
            return a * (1.0f - t) + b * t;
        }

        __global__ void liquidKernel(cudaSurfaceObject_t surface,
                                    int width, int height,
                                    float time,
                                    float mouseX, float mouseY,
                                    float waveFrequency, float waveSpeed,
                                    float waveAmplitude,
                                    float3 backgroundColor, float3 liquidColor,
                                    float liquidAlpha) {

            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            // Normalize edilmiş UV koordinatları
            float u = (float)x / (float)width;
            float v = (float)y / (float)height;

            // En-boy oranını düzelt
            float aspect = (float)width / (float)height;
            u *= aspect;

            // Fare etkileşimi için merkeze doğru kaydırma
            float mouseU = (mouseX / (float)width) * aspect;
            float mouseV = mouseY / (float)height;

            // Fareden uzaklığa göre hafif bir itme (etkileşim)
            float distToMouse = length2(u - mouseU, v - mouseV);
            float mouseInfluence = expf(-distToMouse * 3.0f) * 0.5f;

            // ==========================================
            // AKIŞKAN TÜRBÜLANSI (LİKİT CAM MATEMATİĞİ)
            // ==========================================

            // Başlangıç koordinatları (Frekans ile ölçeklenmiş)
            // Frekans arayüzde 5.0 ile 10.0 arasında çok güzel erimiş cam efekti verir
            float px = u * waveFrequency;
            float py = v * waveFrequency;

            // Zamanı hıza göre ayarla
            float t = time * waveSpeed * 0.1f;

            // Koordinatları kendi içlerinde bük (Space Distortion)
            for (int i = 1; i <= 4; i++) {
                float new_px = px + (1.0f / (float)i) * sinf((float)i * py + t + mouseInfluence);
                float new_py = py + (1.0f / (float)i) * cosf((float)i * px + t + mouseInfluence);
                px = new_px;
                py = new_py;
            }

            // Bükülmüş koordinatlardan bir yoğunluk (intensity) çıkar
            float intensity = cosf(px + py + 1.0f) * 0.5f + 0.5f;

            // ==========================================
            // IŞIK KIRILMASI VE RENKLENDİRME
            // ==========================================

            // Cam parlaması (Specular Highlight) - Yüksek genlikte parlaklık artar
            float highlight = powf(intensity, 4.0f) * waveAmplitude * 10.0f;

            // Renkleri karıştır (Kıvılcım Turuncusu ve Arka Plan)
            float finalR = mix(backgroundColor.x, liquidColor.x, intensity) + highlight;
            float finalG = mix(backgroundColor.y, liquidColor.y, intensity) + highlight;
            float finalB = mix(backgroundColor.z, liquidColor.z, intensity) + highlight;

            // Renklerin patlamasını önlemek için 0-1 arasına sıkıştır (Clamp)
            finalR = fminf(fmaxf(finalR, 0.0f), 1.0f);
            finalG = fminf(fmaxf(finalG, 0.0f), 1.0f);
            finalB = fminf(fmaxf(finalB, 0.0f), 1.0f);

            // (Surface Write)
            float4 pixelColor = make_float4(finalR, finalG, finalB, liquidAlpha);
            surf2Dwrite(pixelColor, surface, x * sizeof(float4), y);
        }

        __global__ void liquidFlowKernel(cudaSurfaceObject_t surface,
                                int width, int height,
                                float time,
                                float flowSpeed, float freq,
                                float3 liquidColor) {

            unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) return;

            // UV koordinatları
            float u = (float)x / (float)width;
            float v = (float)y / (float)height;

            // Likit akış matematiği (Senin gönderdiğin yapıyı 2D'ye uyarladık)
            float mov = time * flowSpeed;

            // 2D uzayda 3 farklı fazlı sinüs dalgası
            float f_x = sinf(u * freq + mov * 0.8f);
            float f_y = sinf(v * freq + mov * 0.4f);
            float f_z = sinf((u + v) * freq + mov * 0.5f); // 3. katman için çapraz dalga

            float sum = f_x + f_y + f_z;
            sum = (sum + 3.0f) / 6.0f; // 0 ile 1 arasına çek

            // Kontrastı artır (Kırılgan/Keskin hatlar için)
            sum = powf(sum, 3.0f);

            // Renkleri ata (Sıvının rengini belirtilen liquidColor ile çarp)
            float r = sum * liquidColor.x;
            float g = sum * liquidColor.y;
            float b = sum * liquidColor.z;

            // Yüzeye yaz
            float4 pixelColor = make_float4(r, g, b, 1.0f);
            surf2Dwrite(pixelColor, surface, x * sizeof(float4), y);
        }

        void launchLiquidFlowShader(cudaSurfaceObject_t surface, int width, int height,
                            float time, float flowSpeed, float freq, float3 liquidColor) {
            dim3 threads(16, 16);
            dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

            liquidFlowKernel<<<blocks, threads>>>(surface, width, height, time, flowSpeed, freq, liquidColor);
            cudaDeviceSynchronize();
        }

        void launchLiquidShader(cudaSurfaceObject_t surface, int width, int height,
                                float time, float mouseX, float mouseY,
                                float waveFrequency, float waveSpeed, float waveAmplitude,
                                float3 backgroundColor, float3 liquidColor, float liquidAlpha) {

            dim3 threads(16, 16);
            dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

            liquidKernel<<<blocks, threads>>>(surface, width, height,
                                              time, mouseX, mouseY,
                                              waveFrequency, waveSpeed, waveAmplitude,
                                              backgroundColor, liquidColor, liquidAlpha);

            cudaDeviceSynchronize();
        }

    }
}