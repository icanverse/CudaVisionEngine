#pragma once
#include <string>
#include <functional>

class RightPanel {
public:
    RightPanel();
    void render(float displayWidth, float displayHeight);

    void setOnImageImportedCallback(std::function<void(const std::string&)> callback) {
        onImageImported = callback;
    }

private:
    std::function<void(const std::string&)> onImageImported;

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

    bool keepOriginalSize;
};