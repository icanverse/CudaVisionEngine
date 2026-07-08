#pragma once
#include <vector>
#include <string>
#include <imgui.h>

#include "Data/ProjectData.h"

// Basit bir proje veri yapısı
struct ProjectItem {
    std::string name;
    std::string imagePath;
    ImTextureID textureID; // İleride gerçek fotoğraf yüklediğinde kullanılacak
};

class LeftPanel {
public:
    void render(float displayWidth, float displayHeight);
    void addProjectToStack(Kivilcim::ProjectData newProject);

private:
    std::vector<Kivilcim::ProjectData> projectStack; // Projeleri tutan stack
    int projectCounter = 1; // "İsimsiz 1", "İsimsiz 2" isimlendirmesi için sayaç
};