#pragma once

namespace Kdata {

    // Bir maskenin temel özelliklerini ve bellek referanslarını tutar
    struct MaskData {
        bool isActive = false;      // Maske şu an devrede mi?
        bool isLinked = true;       // Katman ile birlikte mi hareket edecek? (Sadece Layer Mask için)
        bool isInverted = false;    // Maske tersine mi çevrildi? (Invert)
        
        // --- VRAM VE ÇİZİM VERİLERİ ---
        
        // Arayüzde (ImGui) maskeyi küçük bir önizleme (thumbnail) olarak göstermek için
        unsigned int textureID = 0; 
        
        // CUDA motorunun doğrudan okuyup işlem yapacağı GPU bellek adresi (Genelde 1 kanallı gri tonlamalı veri)
        unsigned char* d_maskData = nullptr; 
        
        // Maskenin o anki sınırları (Bounding Box) - İşlem yükünü hafifletmek için (Sadece seçili alana işlem yapmak)
        int boundsX = 0;
        int boundsY = 0;
        int boundsWidth = 0;
        int boundsHeight = 0;
        
        // Güvenli sıfırlama
        void clear() {
            isActive = false;
            isInverted = false;
            // Not: d_maskData temizliğini motor (Engine) tarafındaki cudaFree ile yapmalısın
        }
    };

}