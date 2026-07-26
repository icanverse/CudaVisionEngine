#include "StbImageSource.h"
#include "stb_image.h" // STB sadece burada çalışacak!
#include <iostream>

StbImageSource::StbImageSource(const char* filepath) : filename(filepath), isLoaded(false) {
    // Görselin boyutlarını önceden okuyalım ki motor hazırlığını yapsın (stbi_info)
    stbi_info(filename, &width, &height, &channels);
}

unsigned char* StbImageSource::grabNextFrame() {
    // Fotoğraf tek bir kareden oluşur. Eğer zaten yüklendiyse nullptr dönerek "Video bitti" mesajı verir.
    if (isLoaded) return nullptr;

    std::cout << "[StbImageSource] Gorsel CPU'ya yukleniyor: " << filename << std::endl;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

    if (!data) {
        std::cerr << "HATA: Gorsel yuklenemedi! " << filename << std::endl;
    }

    isLoaded = true;
    return data; // Veriyi motora teslim et
}

void StbImageSource::releaseFrame(unsigned char* data) {
    if (data) {
        stbi_image_free(data); // STB'nin kendi silme komutu
    }
}