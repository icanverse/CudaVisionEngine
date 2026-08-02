#pragma once
#include <vector>
#include <string>

namespace Kdata {

    enum class UITheme { DARK };
    enum class HARDWARE { CUDA, OPENCL, CPU };

    struct PreferenceData {
        bool isPreferencesChanged = false;

        // Kullanıcı Kaydı
        int userID = 0;
        std::string userName = "Admin";

        UITheme theme = UITheme::DARK;
        std::string language = "tr_TR";

        // Donanım Ayarları
        bool enableHardwareAcceleration = true;
        bool enableHardwareCuda = true;
        bool enableHardwareOpenCL = false;
        bool enableHardwareCPU = false;

        bool enableSharedMemory = true;

        // BELLEK LİMİTLERİ (MB)
        int ram_limit = 8192;
        int vram_limit = 8192;

        // Yollar
        std::vector<std::string> recentProjectsPaths;
        std::string cache_path = "";
        std::string default_export_path = "";

        // Motor içi konfigürasyonlar
        bool enableAutoSave = true;
        int autoSaveIntervalMinutes = 10;

        // ==========================================
        // DONANIM ÖNBELLEĞİ (Sadece 1 Kez Taranıp Buraya Yazılır)
        // ==========================================
        std::string hw_cpuModel = "Bilinmiyor";
        std::string hw_gpuModel = "Bilinmiyor";
        int hw_totalRamMB = 0;
        int hw_totalVramMB = 0;
        int hw_cudaCores = 0;
    };

}