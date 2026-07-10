#pragma once
#include "UI/Data/ProjectData.h"
#include <functional>

#include "Layers/QuickToolbar.h"

class Workspace {
public:
    Workspace();
    void render(float displayWidth, float displayHeight);

    void loadProject(Kivilcim::ProjectData* project);

    void setOnCloseCallback(std::function<void()> callback) {
        onClose = callback;
    }

private:
    Kivilcim::ProjectData* activeProject;
    std::function<void()> onClose;

    // --- ARAYÜZ BİLEŞENLERİ ---
    QuickToolbar quickToolbar; // Sadece toolbarımız var
};
