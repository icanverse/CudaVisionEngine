#pragma once
#include "Data/WorkspaceStateData.h"

class QuickInspectorToolbox {
public:
    QuickInspectorToolbox();
    ~QuickInspectorToolbox();

    // Diğer panellerde olduğu gibi state üzerinden beslenecek
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);

    // Layer paneli ile aynı genişliği kullanması için
    static float getPanelWidth() { return 400.0f; } // Projendeki varsayılan genişlik neyse buraya onu gir
    static float getPanelHeight() { return 375.0f; } // Dinamik panelin yüksekliği
};