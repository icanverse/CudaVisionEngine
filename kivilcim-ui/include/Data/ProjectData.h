#pragma once
#include <string>
#include <vector>
#include <vector_types.h>

namespace Kivilcim {

    struct ProjectData {
        int id;                     // Benzersiz proje ID'si
        std::string name;           // Proje adı (Örn: "Isimsiz 1")
        std::string date;           // Oluşturulma tarihi
        std::string imagePath;      // Orijinal görselin diskteki yolu (Görsel seçilmediyse boş kalır)
        unsigned int textureID;     // VRAM'deki küçük resmin (thumbnail) OpenGL ID'si

        // --- PROJE ŞABLONU VE TUVAL (CANVAS) BİLGİLERİ ---
        bool keepOriginalSize;      // Orijinal görsel boyutları mı korunacak?
        int projectWidth;           // Proje/Tuval Genişliği
        int projectHeight;          // Proje/Tuval Yüksekliği
        int dimMetric;              // Boyut Birimi (0: Piksel, 1: İnç, 2: cm)
        int orientation;            // Oryantasyon (0: Dikey, 1: Yatay)
        int resolution;             // Çözünürlük (DPI/PPI)
        int resMetric;              // Çözünürlük Birimi (0: Px/İnç, 1: Px/cm)
        int bgContentMode;          // Arka Plan Tipi (0: Beyaz, 1: Siyah, 2: Şeffaf, 3: Özel)
        float bgColor[3];           // Özel Arka Plan Rengi (RGB)

        // --- GÖRSEL METADATASI (Sadece görsel yüklendiyse geçerlidir) ---
        int2 size;                  // Orijinal görsel piksel boyutları
        int channels;
        size_t fileSize;

        // VRAM Yönetimi
        float* d_imageData;
        bool isLoadedToGPU;

        // UI
        bool isSelected;
        bool isModified;

        // İşlem Kaydı
        std::vector<std::string> appliedFilters;

        // Materyaller
        std::string kvlcmDir;       // .kvlcm dosyalarını içeren klasör path


        // Constructor (Varsayılan değerlerle ve güvenli başlatma)
        // DİKKAT: p_path artık opsiyonel, görsel seçilmezse boş atanır.
        ProjectData(int p_id, const std::string& p_name, const std::string& p_path = "")
            : id(p_id), name(p_name), imagePath(p_path), textureID(0),
              keepOriginalSize(true), projectWidth(1920), projectHeight(1080),
              dimMetric(0), orientation(1), resolution(72), resMetric(0),
              bgContentMode(0), bgColor{1.0f, 1.0f, 1.0f},
              size{0, 0}, channels(0), fileSize(0),
              d_imageData(nullptr), isLoadedToGPU(false),
              isSelected(false), isModified(false) {}
    };

}