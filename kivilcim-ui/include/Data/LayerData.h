#pragma once
#include <string>
#include <vector>
#include "MaskData.h"

namespace Kdata {

    // Katman Türleri
    enum class LayerType {
        PIXEL = 0,     // Standart görsel/boyama katmanı
        TEXT = 1,      // Yazı katmanı
        ADJUSTMENT = 2 // Ayar katmanı (Parlaklık, Kontrast vb.)
    };

    // Karışım Modları (Blend Modes)
    enum class BlendMode {
        NORMAL = 0,
        MULTIPLY,
        SCREEN,
        OVERLAY
    };

    // Transform (Konum, Ölçek, Döndürme)
    struct Transform {
        float posX = 0.0f;
        float posY = 0.0f;
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float rotation = 0.0f; // Derece cinsinden açı
    };

    struct Layer {
        int id;
        std::string name;

        LayerType type = LayerType::PIXEL;
        BlendMode blendMode = BlendMode::NORMAL;

        bool isVisible = true;
        bool isLocked = false;
        float opacity = 1.0f;

        Transform transform;           // Katmanın tuvaldeki konumu ve boyutu
        std::string sourcePath = "";   // Eğer dışarıdan sürüklenmiş bir PNG ise orijinal yolu

        unsigned int thumbnailTextureID = 0;
        MaskData layerMask;
    };

    struct LayerData {
        std::vector<Layer> layers;
        int activeLayerIndex = -1;

        bool hasActiveLayer() const {
            return activeLayerIndex >= 0 && activeLayerIndex < layers.size();
        }
    };
}