#pragma once
#include <vector>
#include <string>

namespace Kdata {

    enum class UITheme { DARK };
    enum class HARDWARE { CUDA, OPENCL, CPU };

    struct PreferenceData {
        bool isPreferencesChanged = false;

        // Kullanıcı Kaydı
        int userID = 0; // short yerine int yapıldı
        std::string userName = "Admin";

        UITheme theme = UITheme::DARK;
        std::string language = "tr_TR";

        // Donanım Ayarları
        bool enableHardwareAcceleration = true;
        bool enableHardwareCuda = true;
        bool enableHardwareOpenCL = false;
        bool enableHardwareCPU = false;

        bool enableSharedMemory = true;

        // BELLEK LİMİTLERİ İÇİN KESİN ÇÖZÜM
        int ram_limit = 8192;   // unsigned short yerine int yapıldı
        int vram_limit = 8192;  // unsigned short yerine int yapıldı

        // Yollar
        std::vector<std::string> recentProjectsPaths;
        std::string cache_path = "";
        std::string default_export_path = "";

        // Motor içi konfigürasyonlar
        bool enableAutoSave = true;
        int autoSaveIntervalMinutes = 10; // short yerine int yapıldı
    };

}