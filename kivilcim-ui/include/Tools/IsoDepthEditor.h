#pragma once
#include "imgui.h"
#include <vector>

namespace Kivilcim {
    namespace Tools {

        // (0.0f - 1.0f) arası normalize edilmiş resim koordinatı
        struct ControlPoint {
            ImVec2 uv;
        };

        struct IsoLine {
            std::vector<ControlPoint> points;
            float depthValue = 0.5f; // Z ekseni / Derinlik değeri
            ImVec4 color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f); // Arayüzde çizgiyi ayırt etmek için
        };

        class IsoDepthEditor {
        private:
            std::vector<IsoLine> isoLines;
            int selectedLineIndex = -1;

            // Çizim tuvali için yardımcı değişkenler
            ImVec2 canvasMin;
            ImVec2 canvasMax;
            ImVec2 canvasSize;

        public:
            IsoDepthEditor();
            ~IsoDepthEditor() = default;

            bool isOpen = false; // TopPanel'den tetiklenecek bayrak

            // Editörün ana render fonksiyonu
            void render(unsigned int baseTextureID, float imageWidth, float imageHeight);

            // İleride CUDA'ya veri göndermek için getter'lar
            const std::vector<IsoLine>& getIsoLines() const { return isoLines; }
            void clearLines() { isoLines.clear(); selectedLineIndex = -1; }
        };

    } // namespace Tools
} // namespace Kivilcim