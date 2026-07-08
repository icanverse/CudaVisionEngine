#pragma once
#include <string>
#include <vector>
#include <vector_types.h>

namespace Kivilcim {

    struct ProjectData {
        int id;                     // Benzersiz proje ID'si
        std::string name;           // Proje adı (Örn: "Isimsiz 1")
        std::string date;           //
        std::string imagePath;      // Orijinal görselin diskteki yolu
        unsigned int textureID;     // VRAM'deki küçük resmin (thumbnail) OpenGL ID'si

        // image metadata
        int2 size;
        int channels;
        size_t fileSize;

        // vRam Yönetimi
        float* d_imageData;
        bool isLoadedToGPU;

        // uı
        bool isSelected;
        bool isModified;

        // işlem kaydı
        std::vector<std::string> appliedFilters;

        // materyaller için
        std::string kvlcmDir;       // .kvlcm dosyalarını içeren klasör path


        // Constructor (Varsayılan değerlerle başlatmak için)
        // Constructor (Tüm kritik değerleri güvenli bir şekilde başlatmak için)
        // Constructor (Tüm kritik değerleri güvenli bir şekilde başlatmak için)
        ProjectData(int p_id, const std::string& p_name, const std::string& p_path)
            : id(p_id), name(p_name), imagePath(p_path), textureID(0),
              size{0, 0}, channels(0), fileSize(0),
              d_imageData(nullptr), isLoadedToGPU(false),
              isSelected(false), isModified(false) {}
    };

}
