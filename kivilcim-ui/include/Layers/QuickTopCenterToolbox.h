#pragma once

#include "Data/WorkspaceStateData.h"

class QuickTopCenterToolbox {
public:
    QuickTopCenterToolbox() = default;
    ~QuickTopCenterToolbox() = default;

    // YENİ: Sadece state referansı alarak çalışır
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);
};