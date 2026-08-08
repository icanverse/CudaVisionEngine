#pragma once

#include "Data/ProjectData.h"
#include <string>
#include <functional>
#include <atomic>
#include <thread>

class CudaDynamicTexture;

class RightPanel {
public:
    RightPanel();
    ~RightPanel();

    void render(float displayWidth, float displayHeight);

    // MainUI'den gelen callback'i kaydeder
    void setOnProjectCreatedCallback(std::function<void(const Kdata::ProjectData&)> callback) {
        onProjectCreated = callback;
    }

private:
    std::function<void(const Kdata::ProjectData&)> onProjectCreated;

    // --- UI ve Kdata Hazırlık Durumları ---
    char projectNameBuf[128];
    int docWidth, docHeight;
    int dimMetric, orientation, resolution, resMetric;
    int bgContentMode;
    float bgColor[3];
    bool keepOriginalSize;

    std::string selectedImagePath;
    std::string projectSavePath;

    // --- Görsel Öğeler ---
    CudaDynamicTexture* shaderPreviewTexture;
    float flowTime;

    // --- THREAD GÜVENLİĞİ (YENİ MİMARİ) ---
    std::atomic<bool> isProcessingImage{false};
    std::atomic<bool> isImageReadyForGPU{false};
    std::thread workerThread;

    // Thread tarafından yazılıp ana döngü tarafından okunacak ham veriler
    unsigned char* rawResizedData = nullptr;
    int loadedOrigW = 0;
    int loadedOrigH = 0;

    // Arka plan işlemini başlatan yardımcı fonksiyon
    void startImageProcessing(const std::string& path);
};