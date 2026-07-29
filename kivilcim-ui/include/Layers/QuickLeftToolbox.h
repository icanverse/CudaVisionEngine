#ifndef CUDAVISIONENGINE_QUICKTOOLBAR_H
#define CUDAVISIONENGINE_QUICKTOOLBAR_H

#include "Data/WorkspaceStateData.h"

class QuickLeftToolbox {
public:
    QuickLeftToolbox() = default;
    ~QuickLeftToolbox() = default;

    // Render fonksiyonu sadece state referansı alarak çalışır
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);
};

#endif //CUDAVISIONENGINE_QUICKTOOLBAR_H