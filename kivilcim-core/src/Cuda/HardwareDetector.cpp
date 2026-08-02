#include "../../include/Cuda/HardwareDetector.h"
#include <iostream>
#include <thread>
#include <algorithm>
#include <cstring>
#include <cwchar>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>   // __cpuid / __cpuidex intrinsic'leri için
#include <winioctl.h> // Depolama (SSD/HDD) tespiti için IOCTL sabitleri
#include <dxgi1_6.h>  // GPU, VRAM, HDR ve VRR tespiti için (dxgi.h'den daha geniş)
#pragma comment(lib, "dxgi.lib")
#endif

// CUDA Runtime API - host (.cpp) tarafından çağrılabilir, nvcc gerektirmez.
// Projede zaten CUDA Toolkit include path'i tanımlı olduğu için doğrudan dahil ediyoruz.
#include <cuda_runtime.h>

#include <GLFW/glfw3.h>

namespace Kcore {

    namespace {

        // ---------------------------------------------------------------
        // CPU: SIMD komut seti desteği tespiti
        // ---------------------------------------------------------------
        void detectSimdSupport(HardwareInfoData& info) {
#ifdef _WIN32
            int leaf1[4] = { 0 };
            __cpuid(leaf1, 1);
            info.supportsSSE42 = (leaf1[2] & (1 << 20)) != 0;
            info.supportsAVX   = (leaf1[2] & (1 << 28)) != 0;

            int leaf7[4] = { 0 };
            __cpuidex(leaf7, 7, 0);
            info.supportsAVX2    = (leaf7[1] & (1 << 5)) != 0;
            info.supportsAVX512F = (leaf7[1] & (1 << 16)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
            __builtin_cpu_init();
            info.supportsSSE42   = __builtin_cpu_supports("sse4.2");
            info.supportsAVX     = __builtin_cpu_supports("avx");
            info.supportsAVX2    = __builtin_cpu_supports("avx2");
            info.supportsAVX512F = __builtin_cpu_supports("avx512f");
#endif
        }

#ifdef _WIN32
        // ---------------------------------------------------------------
        // CPU: Gerçek fiziksel çekirdek sayısı (P-core/E-core farkında) + Cache boyutları
        // ---------------------------------------------------------------
        void detectCpuTopologyAndCache(HardwareInfoData& info) {
            DWORD bufferSize = 0;
            GetLogicalProcessorInformationEx(RelationAll, nullptr, &bufferSize);
            if (bufferSize == 0) return;

            std::vector<char> buffer(bufferSize);
            if (!GetLogicalProcessorInformationEx(
                    RelationAll,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
                    &bufferSize)) {
                return;
            }

            int physicalCoreCount = 0;
            char* ptr = buffer.data();
            char* end = buffer.data() + bufferSize;

            while (ptr < end) {
                auto* entry = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);

                if (entry->Relationship == RelationProcessorCore) {
                    physicalCoreCount++;
                } else if (entry->Relationship == RelationCache) {
                    const auto& cache = entry->Cache;
                    unsigned int sizeKB = static_cast<unsigned int>(cache.CacheSize / 1024);
                    // Her seviyeden ilk bulunan değeri al (paylaşımlı cache'lerde tekrar sayımını önler)
                    if (cache.Level == 1 && info.l1CacheKB == 0) {
                        info.l1CacheKB = sizeKB;
                    } else if (cache.Level == 2 && info.l2CacheKB == 0) {
                        info.l2CacheKB = sizeKB;
                    } else if (cache.Level == 3 && info.l3CacheKB == 0) {
                        info.l3CacheKB = sizeKB;
                    }
                }

                ptr += entry->Size;
            }

            if (physicalCoreCount > 0) {
                info.physicalCores = physicalCoreCount; // Kaba tahmini gerçek değerle ez
            }
        }

        // ---------------------------------------------------------------
        // Depolama: Birincil diskin SSD/NVMe mi yoksa HDD mi olduğunu tespit et
        // (Seek Penalty yöntemi - WMI'dan daha hafif)
        // ---------------------------------------------------------------
        void detectPrimaryDiskType(HardwareInfoData& info) {
            HANDLE hDevice = CreateFileW(L"\\\\.\\PhysicalDrive0", 0,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (hDevice == INVALID_HANDLE_VALUE) {
                return;
            }

            STORAGE_PROPERTY_QUERY query = {};
            query.PropertyId = StorageDeviceSeekPenaltyProperty;
            query.QueryType = PropertyStandardQuery;

            DEVICE_SEEK_PENALTY_DESCRIPTOR result = {};
            DWORD bytesReturned = 0;

            if (DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY,
                    &query, sizeof(query), &result, sizeof(result), &bytesReturned, nullptr)) {
                info.primaryDiskIsSSD = !result.IncursSeekPenalty;
                info.primaryDiskModel = result.IncursSeekPenalty
                    ? "HDD (Donel Disk - Seek Penalty Var)"
                    : "SSD / NVMe (Seek Penalty Yok)";
            }

            CloseHandle(hDevice);
        }

        // ---------------------------------------------------------------
        // GPU: Çoklu adaptör, VRAM, paylaşımlı bellek, HDR ve VRR tespiti (DXGI)
        // ---------------------------------------------------------------
        void detectGpuAdapters(HardwareInfoData& info) {
            IDXGIFactory1* pFactory = nullptr;
            if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory)) || !pFactory) {
                return;
            }

            // Sistem genelinde tearing/VRR desteği (yaklaşık gösterge)
            IDXGIFactory5* pFactory5 = nullptr;
            if (SUCCEEDED(pFactory->QueryInterface(__uuidof(IDXGIFactory5), (void**)&pFactory5)) && pFactory5) {
                BOOL allowTearing = FALSE;
                if (SUCCEEDED(pFactory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        &allowTearing, sizeof(allowTearing)))) {
                    info.variableRefreshRateSupported = (allowTearing == TRUE);
                }
                pFactory5->Release();
            }

            IDXGIAdapter1* pAdapter = nullptr;
            UINT adapterIndex = 0;
            bool firstAdapterAssigned = false;

            while (pFactory->EnumAdapters1(adapterIndex, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
                DXGI_ADAPTER_DESC1 desc;
                if (SUCCEEDED(pAdapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    info.gpuAdapterCount++;

                    if (!firstAdapterAssigned) {
                        char descString[128] = { 0 };
                        wcstombs(descString, desc.Description, sizeof(descString) - 1);
                        info.gpuModel = std::string(descString);
                        info.dedicatedVRAM = desc.DedicatedVideoMemory / (1024 * 1024);
                        info.sharedSystemMemory = desc.SharedSystemMemory / (1024 * 1024);

                        if (desc.VendorId == 0x10DE) info.gpuVendor = "NVIDIA";
                        else if (desc.VendorId == 0x1002) info.gpuVendor = "AMD";
                        else if (desc.VendorId == 0x8086) info.gpuVendor = "Intel";
                        else info.gpuVendor = "Harici / Diger";

                        // Anlık kullanılabilir VRAM (IDXGIAdapter3)
                        IDXGIAdapter3* pAdapter3 = nullptr;
                        if (SUCCEEDED(pAdapter->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&pAdapter3)) && pAdapter3) {
                            DXGI_QUERY_VIDEO_MEMORY_INFO memInfo;
                            if (SUCCEEDED(pAdapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memInfo))) {
                                unsigned long long budgetMB = memInfo.Budget / (1024 * 1024);
                                unsigned long long usedMB = memInfo.CurrentUsage / (1024 * 1024);
                                info.freeVRAM = (budgetMB > usedMB) ? (budgetMB - usedMB) : 0;
                            }
                            pAdapter3->Release();
                        }

                        // HDR desteği (bu adaptörün ilk çıkışı üzerinden)
                        IDXGIOutput* pOutput = nullptr;
                        if (SUCCEEDED(pAdapter->EnumOutputs(0, &pOutput)) && pOutput) {
                            IDXGIOutput6* pOutput6 = nullptr;
                            if (SUCCEEDED(pOutput->QueryInterface(__uuidof(IDXGIOutput6), (void**)&pOutput6)) && pOutput6) {
                                DXGI_OUTPUT_DESC1 outDesc;
                                if (SUCCEEDED(pOutput6->GetDesc1(&outDesc))) {
                                    info.hdrSupported =
                                        (outDesc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
                                }
                                pOutput6->Release();
                            }
                            pOutput->Release();
                        }

                        firstAdapterAssigned = true;
                    }
                }
                pAdapter->Release();
                adapterIndex++;
            }

            pFactory->Release();
        }
#endif // _WIN32

        // ---------------------------------------------------------------
        // Monitör: Sistemdeki tüm monitörleri listele (GLFW - cross-platform)
        // ---------------------------------------------------------------
        void detectMonitors(HardwareInfoData& info) {
            int monitorCount = 0;
            GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);
            GLFWmonitor* primary = glfwGetPrimaryMonitor();

            for (int i = 0; i < monitorCount; ++i) {
                GLFWmonitor* mon = monitors[i];
                const GLFWvidmode* mode = glfwGetVideoMode(mon);
                if (!mode) continue;

                MonitorInfo m;
                m.width = mode->width;
                m.height = mode->height;
                m.refreshRate = mode->refreshRate;
                m.isPrimary = (mon == primary);

                const char* name = glfwGetMonitorName(mon);
                if (name) m.name = std::string(name);

                info.monitors.push_back(m);
            }
        }

        // ---------------------------------------------------------------
        // CUDA: Compute capability'ye göre SM başına CUDA core sayısı (yaklaşık, mimariye göre)
        // ---------------------------------------------------------------
        int cudaCoresPerSM(int major, int minor) {
            struct SMToCores { int sm; int cores; };
            static const SMToCores table[] = {
                {0x30,192}, {0x32,192}, {0x35,192}, {0x37,192}, // Kepler
                {0x50,128}, {0x52,128}, {0x53,128},             // Maxwell
                {0x60,64},  {0x61,128}, {0x62,128},             // Pascal
                {0x70,64},  {0x72,64},  {0x75,64},              // Volta / Turing
                {0x80,64},  {0x86,128}, {0x87,128}, {0x89,128}, // Ampere / Ada
                {0x90,128},                                     // Hopper
            };
            int smVer = (major << 4) + minor;
            for (const auto& e : table) {
                if (e.sm == smVer) return e.cores;
            }
            return 64; // Bilinmeyen/yeni mimari için makul varsayılan
        }

        // ---------------------------------------------------------------
        // CUDA: Sistemdeki tüm CUDA uyumlu cihazları tespit et
        // ---------------------------------------------------------------
        void detectCudaDevices(HardwareInfoData& info) {
            int deviceCount = 0;
            if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
                info.cudaAvailable = false;
                return;
            }

            info.cudaAvailable = true;

            int previousDevice = 0;
            cudaGetDevice(&previousDevice);

            for (int i = 0; i < deviceCount; ++i) {
                cudaDeviceProp prop{};
                if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;

                CudaDeviceInfo dev;
                dev.deviceId = i;
                dev.name = prop.name;
                dev.computeCapabilityMajor = prop.major;
                dev.computeCapabilityMinor = prop.minor;
                dev.multiprocessorCount = prop.multiProcessorCount;
                dev.cudaCoresPerSM = cudaCoresPerSM(prop.major, prop.minor);
                dev.totalCudaCores = dev.cudaCoresPerSM * dev.multiprocessorCount;
                dev.totalGlobalMemMB = static_cast<unsigned long long>(prop.totalGlobalMem) / (1024 * 1024);
                dev.sharedMemPerBlockKB = static_cast<unsigned long long>(prop.sharedMemPerBlock) / 1024;
                dev.maxThreadsPerBlock = prop.maxThreadsPerBlock;
                dev.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
                dev.warpSize = prop.warpSize;
                dev.unifiedAddressing = prop.unifiedAddressing != 0;

                // Anlık boş VRAM (yalnızca ilgili cihaz "current" yapılınca sorgulanabilir)
                if (cudaSetDevice(i) == cudaSuccess) {
                    size_t freeMem = 0, totalMem = 0;
                    if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
                        dev.freeGlobalMemMB = static_cast<unsigned long long>(freeMem) / (1024 * 1024);
                    }
                }

                info.cudaDevices.push_back(dev);
            }

            cudaSetDevice(previousDevice); // Aktif cihazı eski haline döndür
        }

    } // anonymous namespace

    HardwareInfoData HardwareDetector::inspectSystem(GLFWwindow* window) {
        HardwareInfoData info;

        // 1. CPU BİLGİLERİ (temel)
        info.logicalCores = std::thread::hardware_concurrency();
        if (info.logicalCores <= 0) {
            info.physicalCores = 2; // Tespit edilemedi, güvenli varsayılan
        } else if (info.logicalCores == 1) {
            info.physicalCores = 1;
        } else {
            info.physicalCores = info.logicalCores / 2; // Kaba tahmin, Windows'ta asıl değerle ezilecek
        }

#ifdef _WIN32
        char cpuBrand[0x40] = { 0 }; // Çöp veri kalmaması için sıfırlandı
        int cpuInfo[4] = { 0 };
        __cpuid(cpuInfo, 0x80000000);
        unsigned int nExIds = cpuInfo[0];
        if (nExIds >= 0x80000004) {
            __cpuid(cpuInfo, 0x80000002);
            memcpy(cpuBrand, cpuInfo, sizeof(cpuInfo));
            __cpuid(cpuInfo, 0x80000003);
            memcpy(cpuBrand + 16, cpuInfo, sizeof(cpuInfo));
            __cpuid(cpuInfo, 0x80000004);
            memcpy(cpuBrand + 32, cpuInfo, sizeof(cpuInfo));
            info.cpuModel = std::string(cpuBrand);
        }
#else
        info.cpuModel = "Standart Çok Çekirdekli İşlemci";
#endif

        detectSimdSupport(info);

#ifdef _WIN32
        detectCpuTopologyAndCache(info); // Gerçek fiziksel çekirdek sayısı + cache boyutları
#endif

        // 2. RAM BİLGİLERİ
#ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) {
            info.totalSysRAM = status.ullTotalPhys / (1024 * 1024);     // MB
            info.availableSysRAM = status.ullAvailPhys / (1024 * 1024); // MB
        }
#else
        info.totalSysRAM = 16384; // Güvenli varsayılan
        info.availableSysRAM = 8192;
#endif

        // 3. MONİTÖR VE EKRAN BİLGİLERİ (GLFW Üzerinden)
        if (window != nullptr) {
            int w, h;
            glfwGetWindowSize(window, &w, &h);
            info.screenWidth = w;
            info.screenHeight = h;

            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            if (monitor) {
                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                if (mode) {
                    info.screenWidth = mode->width;
                    info.screenHeight = mode->height;
                    info.refreshRate = mode->refreshRate;
                }
                const char* monName = glfwGetMonitorName(monitor);
                if (monName) {
                    info.monitorName = std::string(monName);
                }
            }
        } else {
            // Pencere yoksa varsayılan birincil monitör modunu al
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            if (monitor) {
                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                if (mode) {
                    info.screenWidth = mode->width;
                    info.screenHeight = mode->height;
                    info.refreshRate = mode->refreshRate;
                }
            }
        }
        detectMonitors(info); // Tüm bağlı monitörlerin listesi

        // 4. GPU VE VRAM BİLGİLERİ
#ifdef _WIN32
        detectGpuAdapters(info); // Çoklu adaptör, VRAM, HDR, VRR (DXGI)
#else
        info.gpuModel = "OpenGL Uyumlu Grafik Kartı";
        info.gpuVendor = "Bilinmiyor";
        info.dedicatedVRAM = 4096;
        info.gpuAdapterCount = 1;
#endif

        // 5. CUDA CİHAZ BİLGİLERİ (cross-platform)
        detectCudaDevices(info);

        // 6. DEPOLAMA BİLGİLERİ
#ifdef _WIN32
        detectPrimaryDiskType(info);
#else
        info.primaryDiskModel = "Bilinmeyen Depolama Birimi (Linux)";
#endif

        return info;
    }

    void HardwareDetector::printHardwareReport(const HardwareInfoData& info) {
        std::cout << "\n================ KIVILCIM HARDWARE RAPORU ================\n";
        std::cout << " [CPU] Model            : " << info.cpuModel << "\n";
        std::cout << " [CPU] Cekirdek (M/F)   : " << info.logicalCores << " Mantiksal / " << info.physicalCores << " Fiziksel\n";
        std::cout << " [CPU] SIMD Destegi     : "
                   << (info.supportsSSE42 ? "SSE4.2 " : "")
                   << (info.supportsAVX ? "AVX " : "")
                   << (info.supportsAVX2 ? "AVX2 " : "")
                   << (info.supportsAVX512F ? "AVX-512F " : "")
                   << ((!info.supportsSSE42 && !info.supportsAVX && !info.supportsAVX2 && !info.supportsAVX512F) ? "Tespit edilemedi" : "")
                   << "\n";
        std::cout << " [CPU] Cache (L1/L2/L3) : " << info.l1CacheKB << " KB / " << info.l2CacheKB << " KB / " << info.l3CacheKB << " KB\n";
        std::cout << " [RAM] Toplam / Bos     : " << info.totalSysRAM << " MB / " << info.availableSysRAM << " MB\n";
        std::cout << " [GPU] Adaptor Sayisi   : " << info.gpuAdapterCount << "\n";
        std::cout << " [GPU] Uretici (Vendor) : " << info.gpuVendor << "\n";
        std::cout << " [GPU] Model            : " << info.gpuModel << "\n";
        std::cout << " [GPU] Ayrilmis VRAM    : " << info.dedicatedVRAM << " MB\n";
        std::cout << " [GPU] Tahmini Bos VRAM : " << info.freeVRAM << " MB\n";
        std::cout << " [GPU] Paylasimli Bellek: " << info.sharedSystemMemory << " MB\n";
        std::cout << " [GPU] HDR / VRR        : " << (info.hdrSupported ? "HDR Var" : "HDR Yok")
                   << " / " << (info.variableRefreshRateSupported ? "VRR Var" : "VRR Yok") << "\n";

        if (info.cudaAvailable) {
            std::cout << " [CUDA] Bulunan Cihaz   : " << info.cudaDevices.size() << "\n";
            for (const auto& dev : info.cudaDevices) {
                std::cout << "    -> [" << dev.deviceId << "] " << dev.name
                           << " | SM " << dev.computeCapabilityMajor << "." << dev.computeCapabilityMinor
                           << " | " << dev.multiprocessorCount << " SM x " << dev.cudaCoresPerSM
                           << " core = " << dev.totalCudaCores << " CUDA core"
                           << " | VRAM " << dev.freeGlobalMemMB << "/" << dev.totalGlobalMemMB << " MB bos\n";
            }
        } else {
            std::cout << " [CUDA] Durum           : CUDA uyumlu cihaz bulunamadi\n";
        }

        std::cout << " [MONITOR] Birincil     : " << info.screenWidth << "x" << info.screenHeight
                   << " @ " << info.refreshRate << "Hz (" << info.monitorName << ")\n";
        if (info.monitors.size() > 1) {
            std::cout << " [MONITOR] Toplam       : " << info.monitors.size() << " monitor tespit edildi\n";
        }
        std::cout << " [DEPOLAMA] Birincil    : " << info.primaryDiskModel
                   << " (" << (info.primaryDiskIsSSD ? "SSD/NVMe" : "HDD") << ")\n";
        std::cout << "==========================================================\n\n";
    }
}