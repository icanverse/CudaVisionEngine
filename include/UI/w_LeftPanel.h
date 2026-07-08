#pragma once
#include <vector>
#include <string>
#include <imgui.h>

// Basit bir proje veri yapısı
struct ProjectItem {
    std::string name;
    std::string imagePath;
    ImTextureID textureID; // İleride gerçek fotoğraf yüklediğinde kullanılacak
};

class LeftPanel {
public:
    void render(float displayWidth, float displayHeight);
    void addPhotoToStack(const std::string& photoPath); // Yeni proje ekleme fonksiyonu

private:
    std::vector<ProjectItem> photoStack; // Projeleri tutan stack
    int projectCounter = 1; // "İsimsiz 1", "İsimsiz 2" isimlendirmesi için sayaç
};