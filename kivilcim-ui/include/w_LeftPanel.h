#pragma once
#include <vector>
#include <string>
#include <functional> // YENİ: Callback için gerekli
#include <imgui.h>

#include "Data/ProjectData.h"

class LeftPanel {
public:
    void render(float displayWidth, float displayHeight);

    void addProjectToStack(Kdata::ProjectData newProject);
    void loadWorkspace();
    void saveWorkspace();

    const std::vector<Kdata::ProjectData>& getProjectStack() const { return projectStack; }

    // YENİ: İstenen ID'ye sahip projenin referansını döndüren yardımcı
    Kdata::ProjectData* getProjectByID(int id) {
        for (auto& p : projectStack) {
            if (p.id == id) return &p;
        }
        return nullptr;
    }

    // YENİ: Çift tıklama sinyalini dışarı aktaran fonksiyon
    void setOnProjectDoubleClickedCallback(std::function<void(int)> callback) {
        onProjectDoubleClicked = callback;
    }

private:
    std::vector<Kdata::ProjectData> projectStack;
    int projectCounter = 1;

    // YENİ: Sinyal değişkeni
    std::function<void(int)> onProjectDoubleClicked;
};