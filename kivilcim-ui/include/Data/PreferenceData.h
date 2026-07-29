#pragma once
#include <vector>
#include <string>

using namespace std;

namespace Kdata {

    enum class UITheme { DARK };
    enum class HARDWARE { CUDA, OPENCL, CPU };

    struct PreferenceData {
        bool isPreferencesChanged = false;

        // Kullanıcı Kaydı
        short userID = 0;
        string userName = "Admin";

        UITheme theme = UITheme::DARK;
        string language = "tr_TR";

        // Donanım Ayarları
        bool enableHardwareAcceleration = true; // CUDA ivmelendirmesi aktif/pasif
        bool enableHardwareCuda = true;
        bool enableHardwareOpenCL = false;
        bool enableHardwareCPU = false;

        bool enableSharedMemory = true;

        short ram_limit = 8;
        short vram_limit = 6;

        // Yollar
        vector<string> recentProjectsPaths;
        string cache_path = "";
        string default_export_path = "";


        // Motor içi konfigürasyonlar
        bool enableAutoSave = true;
        short autoSaveIntervalMinutes = 10;

    };

}