#include "AssetsManager/IconManager.h"
#include "TextureUtility/TextureUtility.h"
#include <iostream>

#include "imgui_impl_opengl3_loader.h"

std::unordered_map<Icon, unsigned int> IconManager::iconCache;

void IconManager::Initialize() {
    std::string basePath = "C:/Users/Can/CLionProjects/CudVisionEngineX/lib-assets/icons/";
    std::string baseIconPath = "base/";
    std::string brushbarPath = "brushbar/";
    std::string elemanteryOperationPath = "elemanteryOperation/";
    std::string layersPath = "layers/";
    std::string leftToolboxPath = "leftToolbox/";
    std::string shapesPath = "shapes/";
    std::string textbarPath = "textbar/";
    std::string topToolboxPath = "topToolbox/";

    // ==========================================
    // BAZ İKONLAR
    // ==========================================
    RegisterIcon(Icon::Copy,             basePath + baseIconPath + "copy.png");
    RegisterIcon(Icon::Delete,           basePath + baseIconPath + "delete.png");
    RegisterIcon(Icon::Folder,           basePath + baseIconPath + "folder.png");
    RegisterIcon(Icon::Download,         basePath + baseIconPath + "download.png");
    RegisterIcon(Icon::Export,           basePath + baseIconPath + "export.png");
    RegisterIcon(Icon::File,             basePath + baseIconPath + "file.png");
    RegisterIcon(Icon::Horizontal_3_Dot, basePath + baseIconPath + "hori3dot.png");
    RegisterIcon(Icon::Import,           basePath + baseIconPath + "import.png");
    RegisterIcon(Icon::Recently,         basePath + baseIconPath + "recently.png");
    RegisterIcon(Icon::Save,             basePath + baseIconPath + "save.png");
    RegisterIcon(Icon::Oversave,         basePath + baseIconPath + "oversave.png");
    RegisterIcon(Icon::Share,            basePath + baseIconPath + "share.png");
    RegisterIcon(Icon::Star,             basePath + baseIconPath + "star.png");
    RegisterIcon(Icon::Upload,           basePath + baseIconPath + "upload.png");
    RegisterIcon(Icon::Vertical_3_Dot,   basePath + baseIconPath + "ver3dot.png");

    // ==========================================
    // ÜST ARAÇ PANELİ
    // ==========================================
    // Baz Bar
    RegisterIcon(Icon::Undo,              basePath + topToolboxPath + "undo.png");
    RegisterIcon(Icon::Redo,              basePath + topToolboxPath + "redo.png");
    RegisterIcon(Icon::Turn_Left,         basePath + topToolboxPath + "turn_left.png");
    RegisterIcon(Icon::Turn_Right,        basePath + topToolboxPath + "turn_right.png");
    RegisterIcon(Icon::Zoom_In,           basePath + topToolboxPath + "zoom_in.png");
    RegisterIcon(Icon::Zoom_Out,          basePath + topToolboxPath + "zoom_out.png");
    RegisterIcon(Icon::Mirror_Horizontal, basePath + topToolboxPath + "mirror_horizontal.png");
    RegisterIcon(Icon::Mirror_Vertical,   basePath + topToolboxPath + "mirror_vertical.png");

    // Şekil Bar
    RegisterIcon(Icon::Circle,            basePath + shapesPath + "circle.png");
    RegisterIcon(Icon::Line,              basePath + shapesPath + "line.png");
    RegisterIcon(Icon::Square,            basePath + shapesPath + "square.png");
    RegisterIcon(Icon::Vector,            basePath + shapesPath + "vector.png");

    // Çizim & Metin Bar
    RegisterIcon(Icon::Erase,             basePath + brushbarPath + "erase.png");
    RegisterIcon(Icon::Text_Size,         basePath + textbarPath + "text_size.png");
    RegisterIcon(Icon::Text_Color,        basePath + textbarPath + "text_color.png");

    // ==========================================
    // SOL ARAÇ PANELİ
    // ==========================================
    RegisterIcon(Icon::Brush,                   basePath + leftToolboxPath + "brush.png");
    RegisterIcon(Icon::Color,                   basePath + leftToolboxPath + "color.png");
    RegisterIcon(Icon::Crop,                    basePath + leftToolboxPath + "crop.png");
    RegisterIcon(Icon::Move,                    basePath + leftToolboxPath + "move.png");
    RegisterIcon(Icon::Select_Region_Free,      basePath + leftToolboxPath + "select_region_free.png");
    RegisterIcon(Icon::Select_Region_Rectangle, basePath + leftToolboxPath + "select_region_rectangle.png");
    RegisterIcon(Icon::Text,                    basePath + leftToolboxPath + "text.png");

    // ==========================================
    // SAĞ ARAÇ PANELİ
    // ==========================================
    RegisterIcon(Icon::Contrast,          basePath + elemanteryOperationPath + "contrast.png");
    RegisterIcon(Icon::Temperature,       basePath + elemanteryOperationPath + "temperature.png");

    // ==========================================
    // KATMANLAR PANELİ
    // ==========================================
    RegisterIcon(Icon::Layers_toDown,     basePath + layersPath + "layer_todown.png");
    RegisterIcon(Icon::Layers,            basePath + layersPath + "layers.png");
    RegisterIcon(Icon::Layers_Add,        basePath + layersPath + "layers_add.png");
    RegisterIcon(Icon::Lock,              basePath + layersPath + "lock.png");
    RegisterIcon(Icon::Union,             basePath + layersPath + "union.png");
    RegisterIcon(Icon::Visible,           basePath + layersPath + "visible.png");

    std::cout << "[IconManager] Tum arayuz ikonlari VRAM'e basariyla onbelleklendi.\n";
}

void IconManager::RegisterIcon(Icon id, const std::string& filepath) {
    unsigned int textureID = TextureUtility::LoadTextureFromFile(filepath.c_str());
    if (textureID != 0) {
        iconCache[id] = textureID;
    } else {
        std::cout << "[IconManager - HATA] Ikon yuklenemedi veya bulunamadi: " << filepath << "\n";
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