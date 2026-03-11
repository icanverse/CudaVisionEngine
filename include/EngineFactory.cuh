//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_ENGINEMANAGEMENT_H
#define CUDAVISIONENGINE_ENGINEMANAGEMENT_H

#include <utility> // std::swap için eklendi
#include <cuda_runtime.h>

class EngineFactory {
private:
    // Görsel Özellikleri
    int width;
    int height;
    int channels;
    size_t totalElementCount; // w * h * c

    // Pointerlar
    float* d_data;      // Device (GPU) - İşlenmiş Float Veri (0.0 - 1.0 arası)
    float* d_temp_data; // Çift Bellek Mimarisi için geçici VRAM alanı

    // Yardımcı Fonksiyonlar
    void allocateMemory();
    void cleanUp();

public:
    // YENİ: Artık dosya adı yok. Motor sadece boyutları alıp VRAM'de yer ayırır.
    EngineFactory(int w, int h, int c);

    // Destructor: Belleği temizler
    ~EngineFactory();

    // YENİ: RAM'den VRAM'e veri pompalar (Char -> Float normalizasyonu yapar)
    EngineFactory& uploadFrame(const unsigned char* cpu_data);

    // YENİ: VRAM'den RAM'e işlenmiş veriyi çeker (Float -> Char denormalizasyonu yapar)
    void downloadFrame(unsigned char* cpu_data);

    // Getterlar
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    float* getDeviceData() const { return d_data; }

    void updateDeviceData(const float* newData) {
        if (d_data && newData) {
            size_t size = width * height * channels * sizeof(float);
            cudaMemcpy(d_data, newData, size, cudaMemcpyDeviceToDevice);
        }
    }

    // YENİ: Veriyi CPU'ya indirmeden, doğrudan başka bir VRAM adresine (Interop için) yazar
    void copyToDeviceUchar(unsigned char* d_dest_uchar);

    // Renk Uzayı Dönüşümleri (Ping-Pong kullanır)
    EngineFactory& rgbToHsv();
    EngineFactory& hsvToRgb();

    // Filtreler (In-place çalışır)
    EngineFactory& applyTemperature(float temperature);
    EngineFactory& applyShadowsHighlights(float shadowAmount, float highlightAmount);
    EngineFactory& applyGamma(float gamma);
};

#endif //CUDAVISIONENGINE_ENGINEMANAGEMENT_H