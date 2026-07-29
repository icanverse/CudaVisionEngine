#pragma once
#include <string>
#include <functional>
#include <atomic>
#include "Data/ProjectData.h"

// İleri bildirim (Incomplete type hatasını önlemek için)
class CudaDynamicTexture;

class RightPanel {
public:
    RightPanel();
    ~RightPanel(); // Bellek sızıntısını önleyecek destructor
    void render(float displayWidth, float displayHeight);

    void setOnImageImportedCallback(std::function<void(const std::string&)> callback) {
        onImageImported = callback;
    }

    // YENİ VE DOĞRU TANIMLAMA (Inline olarak set ediliyor)
    void setOnProjectCreatedCallback(std::function<void(const Kdata::ProjectData&)> callback) {
        onProjectCreated = callback;
    }

private:
    std::function<void(const std::string&)> onImageImported;

    // EKSİK OLAN DEĞİŞKEN BURAYA EKLENDİ
    std::function<void(const Kdata::ProjectData&)> onProjectCreated;

    // --- UI DURUM (STATE) DEĞİŞKENLERİ ---
    char projectNameBuf[128];

    int docWidth;                 // Genişlik
    int docHeight;                // Yükseklik
    int dimMetric;                // 0: Piksel, 1: İnç, 2: cm

    int orientation;              // 0: Dikey (Portrait), 1: Yatay (Landscape)

    int resolution;               // Çözünürlük (DPI/PPI)
    int resMetric;                // 0: Piksel/İnç, 1: Piksel/cm

    int bgContentMode;            // 0: Beyaz, 1: Siyah, 2: Şeffaf, 3: Özel
    float bgColor[3];             // Arka Plan Rengi

    std::string selectedImagePath;
    std::string projectSavePath;  // Proje Kayıt Klasörü

    bool keepOriginalSize;

    // --- ASENKRON YÜKLEME (THREAD) DEĞİŞKENLERİ ---
    std::atomic<bool> isProcessingImage{false};
    std::atomic<bool> isImageReadyForGPU{false};

    unsigned char* rawResizedData = nullptr;
    int loadedOrigW = 0;
    int loadedOrigH = 0;

    // --- LİKİT AKIŞ (SHADER) DEĞİŞKENLERİ ---
    CudaDynamicTexture* shaderPreviewTexture;
    float flowTime;
};