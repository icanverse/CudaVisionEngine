//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_ENGINEMANAGEMENT_H
#define CUDAVISIONENGINE_ENGINEMANAGEMENT_H

class EngineFactory {
private:
    // Görsel Özellikleri
    int width;
    int height;
    int channels;
    size_t totalElementCount; // w * h * c

    // Pointerlar
    float* d_data; // Device (GPU) - İşlenmiş Float Veri (0.0 - 1.0 arası)

    // Yardımcı Fonksiyonlar
    void allocateMemory();
    void cleanUp();

public:
    // Constructor: Dosya adını alır, yükler ve GPU'ya atar
    EngineFactory(const char* filename);

    // Destructor: Belleği temizler
    ~EngineFactory();

    // Resmi diske kaydeder
    void saveImage(const char* filename);

    // Getterlar (Gerekirse dışarıdan erişim için)
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

};


#endif //CUDAVISIONENGINE_ENGINEMANAGEMENT_H