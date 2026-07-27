#pragma once
#include <string>
#include <vector>

#include "MaskData.h"

namespace Kdata {

    struct Layer {
        int id;
        std::string name;
        bool isVisible = true;
        bool isLocked = false;
        float opacity = 1.0f;
        unsigned int textureID = 0; // Katmanın VRAM'deki OpenGL doku karşılığı
        MaskData layerMask;
    };

    struct LayerData {
        std::vector<Layer> layers;
        int activeLayerIndex = -1; // Seçili katman (hiçbiri seçili değilse -1)

        // Hızlı erişim ve güvenlik fonksiyonları eklenebilir
        bool hasActiveLayer() const {
            return activeLayerIndex >= 0 && activeLayerIndex < layers.size();
        }
    };

}
