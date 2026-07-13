#pragma once
#include "UI/Data/ProjectData.h"
#include <functional>

#include "Layers/QuickLeftToolbox.h"
#include "Layers/QuickTopRightToolbox.h"

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
    QuickLeftToolbox quickToolbar; // Sadece toolbarımız var
    QuickTopRightToolbox topToolbox;
};
