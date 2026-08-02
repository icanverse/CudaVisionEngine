#pragma once

#include "AssetsManager/IconManager.h"
#include "Data/WorkspaceStateData.h" // YENİ: Tek Gerçek Kaynak

#include <string>
#include <vector>

class QuickLayersToolbox {
public:
    QuickLayersToolbox();
    ~QuickLayersToolbox();

    // YENİ: Doğrudan state'i alır.
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);

    static float getPanelWidth() { return 400.0f; }
    static float getPanelTop() { return 200.0f; }

private:
    struct Tool {
        Kdata::InstantAction actionId; // Kdata mimarisindeki InstantAction'ı kullanıyoruz
        std::string name;
        Icon icon;
        std::string tooltip;
    };

    void renderToolbar(Kdata::WorkspaceStateData* state, float iconSide, float spacing);
    void renderLayerList(Kdata::WorkspaceStateData* state);
    void renderLayerRow(Kdata::WorkspaceStateData* state, Kdata::Layer& layer); // Kdata::Layer referansı kullanılıyor

    std::vector<Tool> availableTools;
};