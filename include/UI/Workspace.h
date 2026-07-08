#pragma once
#include "UI/Data/ProjectData.h"
#include <functional>
#include <string>

class Workspace {
public:
    Workspace();
    void render(float displayWidth, float displayHeight);
    
    // Projeyi çalışma alanına yüklemek için
    void loadProject(Kivilcim::ProjectData* project);

    // Ana ekrana geri dönmek için kullanılacak sinyal
    void setOnCloseCallback(std::function<void()> callback) {
        onClose = callback;
    }

private:
    Kivilcim::ProjectData* activeProject;
    std::function<void()> onClose;

    // Alt Panellerin Çizim Fonksiyonları
    void renderTopMenu();
    void renderToolbox(float displayHeight, float menuHeight);
    void renderRightPanels(float displayWidth, float displayHeight, float menuHeight);
    void renderCanvas(float displayWidth, float displayHeight, float menuHeight);
    
    // UI Değişkenleri
    float toolboxWidth = 60.0f;
    float rightPanelWidth = 300.0f;
    int selectedTool = 0; // 0: Taşı, 1: Seçim, 2: Fırça, 3: Silgi vb.
};