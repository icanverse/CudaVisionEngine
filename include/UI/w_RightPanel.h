#pragma once
#include <string>
#include <functional> // std::function için gerekli

class RightPanel {
public:
    void render(float displayWidth, float displayHeight);

    // MainUI'nin bu panele bir görev atamasını sağlayacak setter fonksiyonu
    void setOnImageImportedCallback(std::function<void(const std::string&)> callback) {
        onImageImported = callback;
    }

private:
    // Görsel yüklendiğinde tetiklenecek fonksiyon
    std::function<void(const std::string&)> onImageImported;
};