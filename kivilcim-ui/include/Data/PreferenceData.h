#pragma once
#include <string>

namespace Kdata {

    enum class UITheme { DARK, LIGHT, CUSTOM };

    struct PreferenceData {
        UITheme theme = UITheme::DARK;
        std::string language = "tr_TR";
        
        bool enableHardwareAcceleration = true; // CUDA ivmelendirmesi aktif/pasif
        int defaultExportFormat = 0;            // 0: PNG, 1: JPG
    };

}