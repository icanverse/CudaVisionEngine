#pragma once

#include "Data/WorkspaceStateData.h"

class QuickTopRightToolbox {
public:
    QuickTopRightToolbox() = default;
    ~QuickTopRightToolbox() = default;

    // YENİ: Yalnızca state referansı alarak çalışır
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);
};