#pragma once
#include <cstddef>

namespace Kdata {

    struct ResourceData {
        bool isCudaInitialized = false; // CUDA motorunun durumu
        
        size_t vramUsageBytes = 0;      // Anlık VRAM kullanımı
        size_t vramTotalBytes = 0;      // Ekran kartının toplam VRAM'i
        
        int activeFontIndex = 0;        // ImGui font listesindeki aktif indeks
    };

}