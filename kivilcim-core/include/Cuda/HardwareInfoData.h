#pragma once
#include <string>
#include <vector>

namespace Kcore {

    // Sistemdeki tek bir monitörün bilgilerini tutar (çoklu monitör desteği)
    struct MonitorInfo {
        std::string name = "Bilinmeyen Monitor";
        int width = 1920;
        int height = 1080;
        int refreshRate = 60;
        bool isPrimary = false;
    };

    // Tek bir CUDA uyumlu cihazın (GPU) detaylı bilgilerini tutar
    struct CudaDeviceInfo {
        int deviceId = -1;
        std::string name = "Bilinmeyen CUDA Cihazi";
        int computeCapabilityMajor = 0;
        int computeCapabilityMinor = 0;
        int multiprocessorCount = 0;
        int cudaCoresPerSM = 0;
        int totalCudaCores = 0;
        unsigned long long totalGlobalMemMB = 0;
        unsigned long long freeGlobalMemMB = 0;
        unsigned long long sharedMemPerBlockKB = 0;
        int maxThreadsPerBlock = 0;
        int maxThreadsPerMultiProcessor = 0;
        int warpSize = 0;
        bool unifiedAddressing = false;
    };

    struct HardwareInfoData {
        // ---- CPU Bilgileri ----
        std::string cpuModel = "Bilinmeyen CPU";
        int physicalCores = 0;
        int logicalCores = 0;

        // CPU SIMD Komut Seti Desteği (performans-kritik kod yolu seçimi için)
        bool supportsSSE42 = false;
        bool supportsAVX = false;
        bool supportsAVX2 = false;
        bool supportsAVX512F = false;

        // CPU Cache Bilgileri (KB cinsinden, çekirdek başına; tespit edilemezse 0)
        unsigned int l1CacheKB = 0;
        unsigned int l2CacheKB = 0;
        unsigned int l3CacheKB = 0; // Genelde paylaşımlı (toplam) L3

        // ---- RAM Bilgileri (MB cinsinden) ----
        unsigned long long totalSysRAM = 0;
        unsigned long long availableSysRAM = 0;

        // ---- Ekran / Monitör Bilgileri ----
        // Birincil monitör (geriye dönük uyumluluk için ayrı alanlar)
        int screenWidth = 1920;
        int screenHeight = 1080;
        int refreshRate = 60;
        std::string monitorName = "Ana Monitor";
        // Sistemdeki tüm monitörler
        std::vector<MonitorInfo> monitors;
        bool variableRefreshRateSupported = false; // G-Sync/FreeSync yaklaşık göstergesi (tearing desteği üzerinden)
        bool hdrSupported = false;

        // ---- GPU Bilgileri (DXGI - güvenli, çökme riski olmayan tespit) ----
        std::string gpuVendor = "Genel / Uyumlu GPU";
        std::string gpuModel = "Standart Grafik Birimi";
        unsigned long long dedicatedVRAM = 0;       // MB, birincil adaptör
        unsigned long long freeVRAM = 0;             // MB, çalışma anındaki tahmini boş VRAM
        unsigned long long sharedSystemMemory = 0;   // MB, entegre GPU'larda önemli
        int gpuAdapterCount = 0;                     // Donanımsal (yazılım olmayan) adaptör sayısı

        // ---- CUDA Cihaz Bilgileri ----
        bool cudaAvailable = false;
        std::vector<CudaDeviceInfo> cudaDevices;

        // ---- Depolama Bilgileri ----
        bool primaryDiskIsSSD = false;
        std::string primaryDiskModel = "Bilinmeyen Depolama Birimi";
    };

}