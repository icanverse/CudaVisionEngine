#include "io/AssetsManager/IconManager.h"

#include "io/TextureUtility/TextureUtility.h"
#include <iostream>
#include "imgui_impl_opengl3_loader.h"

std::unordered_map<Icon, unsigned int> IconManager::iconCache;

void IconManager::Initialize() {
    // BURAYA KENDİ PROJE YOLLARINI YAZACAKSIN
    std::string basePath = "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/icons/";

    // YENİ: RegisterIcon olarak çağırıyoruz
    RegisterIcon(Icon::Select, basePath + "select.png");
    RegisterIcon(Icon::Crop,   basePath + "crop.png");
    RegisterIcon(Icon::Pan,    basePath + "pan.png");
    RegisterIcon(Icon::Save,   basePath + "save.png");

    std::cout << "[IconManager] Arayuz ikonlari VRAM'e basariyla onbelleklendi.\n";
}

// YENİ: RegisterIcon olarak tanımlıyoruz
void IconManager::RegisterIcon(Icon id, const std::string& filepath) {
    unsigned int textureID = TextureUtility::LoadTextureFromFile(filepath.c_str());
    if (textureID != 0) {
        iconCache[id] = textureID;
    } else {
        std::cout << "[IconManager - HATA] Ikon yuklenemedi: " << filepath << "\n";
    }
}

unsigned int IconManager::Get(Icon id) {
    auto it = iconCache.find(id);
    if (it != iconCache.end()) {
        return it->second;
    }
    return 0;
}

void IconManager::Shutdown() {
    for (auto const& [id, texID] : iconCache) {
        glDeleteTextures(1, &texID);
    }
    iconCache.clear();
    std::cout << "[IconManager] VRAM ikon onbellekleri temizlendi.\n";
}