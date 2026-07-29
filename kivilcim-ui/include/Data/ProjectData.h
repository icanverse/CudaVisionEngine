#pragma once
#include <string>
#include <vector>
#include <vector_types.h>

namespace Kdata {

    struct ProjectData {
        int id;                     // Benzersiz proje ID'si
        std::string name;           // Proje adı (Örn: "Isimsiz 1")[cite: 9]
        std::string date;           // Oluşturulma tarihi[cite: 9]
        std::string imagePath;      // Orijinal görselin diskteki yolu (Görsel seçilmediyse boş kalır)[cite: 9]
        unsigned int proxyID;       // VRAM'deki resimin orijianlinden küçük hali.[cite: 9]
        unsigned int textureID;     // VRAM'deki küçük resmin (thumbnail) OpenGL ID'si[cite: 9]

        // --- PROJE ŞABLONU VE TUVAL (CANVAS) BİLGİLERİ ---
        bool keepOriginalSize;      // Orijinal görsel boyutları mı korunacak?[cite: 9]
        int projectWidth;           // Proje/Tuval Genişliği[cite: 9]
        int projectHeight;          // Proje/Tuval Yüksekliği[cite: 9]
        int dimMetric;              // Boyut Birimi (0: Piksel, 1: İnç, 2: cm)[cite: 9]
        int orientation;            // Oryantasyon (0: Dikey, 1: Yatay)[cite: 9]
        int resolution;             // Çözünürlük (DPI/PPI)[cite: 9]
        int resMetric;              // Çözünürlük Birimi (0: Px/İnç, 1: Px/cm)[cite: 9]
        int bgContentMode;          // Arka Plan Tipi (0: Beyaz, 1: Siyah, 2: Şeffaf, 3: Özel)[cite: 9]
        float bgColor[3];           // Özel Arka Plan Rengi (RGB)[cite: 9]

        // --- GÖRSEL METADATASI (Sadece görsel yüklendiyse geçerlidir) ---
        int2 size;                  // Orijinal görsel piksel boyutları[cite: 9]
        int channels;               //[cite: 9]
        size_t fileSize;            //[cite: 9]

        // UI
        bool isSelected;            //[cite: 9]
        bool isModified;            //[cite: 9]

        // Materyaller
        std::string kvlcmDir;       // .kvlcm dosyalarını içeren klasör path[cite: 9]


        // Constructor (Varsayılan değerlerle ve güvenli başlatma)[cite: 9]
        // DİKKAT: p_path artık opsiyonel, görsel seçilmezse boş atanır.[cite: 9]
        ProjectData(int p_id, const std::string& p_name, const std::string& p_path = "")
            : id(p_id), name(p_name), imagePath(p_path), proxyID(0), textureID(0),
              keepOriginalSize(true), projectWidth(1920), projectHeight(1080),
              dimMetric(0), orientation(1), resolution(72), resMetric(0),
              bgContentMode(0), bgColor{1.0f, 1.0f, 1.0f},
              size{0, 0}, channels(0), fileSize(0),
              isSelected(false), isModified(false) {}
    };

}