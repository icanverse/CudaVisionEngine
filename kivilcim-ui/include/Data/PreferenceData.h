#pragma once
#include <vector>
#include <string>

namespace Kdata {

    enum class UITheme { DARK };
    enum class HARDWARE { CUDA, OPENCL, CPU };

    struct PreferenceData {
        bool isPreferencesChanged = false;

        // Kullanıcı Kaydı
        short userID = 0;
        std::string userName = "Admin";

        UITheme theme = UITheme::DARK;
        std::string language = "tr_TR";

        // Donanım Ayarları
        bool enableHardwareAcceleration = true; // CUDA ivmelendirmesi aktif/pasif
        bool enableHardwareCuda = true;
        bool enableHardwareOpenCL = false;
        bool enableHardwareCPU = false;

        bool enableSharedMemory = true;

        short ram_limit = 8;
        short vram_limit = 6;

        // Yollar
        std::vector<std::string> recentProjectsPaths;
        std::string cache_path = "";
        std::string default_export_path = "";


        // Motor içi konfigürasyonlar
        bool enableAutoSave = true;
        short autoSaveIntervalMinutes = 10;

    };

}