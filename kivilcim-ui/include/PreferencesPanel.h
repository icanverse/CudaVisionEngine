#pragma once

#include "Data/PreferenceData.h"
#include "TextureUtility/CudaDynamicTexture.cuh"

namespace Kivilcim {
    namespace UI {

        class PreferencesPanel {
        public:
            PreferencesPanel();
            ~PreferencesPanel();

            // Pencerenin açık/kapalı durumunu ve kullanıcı ayarlarını referans olarak alır
            void render(bool& isOpen, Kdata::PreferenceData& userPrefs);

        private:
            CudaDynamicTexture* bgShaderTexture;
            float flowTime;
        };

    }
}