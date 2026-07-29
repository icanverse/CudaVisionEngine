#pragma once
#include "Data/WorkspaceStateData.h" // YENİ: Tek Gerçek Kaynak
#include <functional>

#include "Layers/QuickLeftToolbox.h"
#include "Layers/QuickTopRightToolbox.h"
#include "Layers/WorkspaceToolboxes.h"
#include "WorkspaceTopPanel.h"
#include "tools/IsoDepthEditor.h"

class Workspace {
public:
    Workspace();

    // YENİ: Artık sadece state pointer'ı alarak çalışıyor
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);

    void setOnCloseCallback(std::function<void()> callback) {
        onClose = callback;
    }

private:
    std::function<void()> onClose;

    // --- ARAYÜZ BİLEŞENLERİ (Alt paneller) ---
    QuickLeftToolbox quickToolbar;
    QuickTopRightToolbox topToolbox;
    WorkspaceToolboxes additionalToolboxes;
    WorkspaceTopPanel workspaceTopPanel;

    // --- Araçlar ---
    Kivilcim::Tools::IsoDepthEditor isoEditor;
};