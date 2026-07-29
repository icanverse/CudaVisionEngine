#pragma once

#include "Data/WorkspaceStateData.h"

struct GLFWwindow;

// Yalnizca ayrik Workspace viewport'u icin kullanilir. Ana pencerenin
// TopPanel sinifindan bagimsizdir; bu nedenle mevcut panel davranisini degistirmez.
class WorkspaceTopPanel {
public:
    // YENİ MİMARİ: Artık doğrudan state'i parametre olarak alıyoruz.
    bool render(Kdata::WorkspaceStateData* state, GLFWwindow* window, float displayWidth, unsigned int logoTextureId = 0);

    static float getPanelHeight();
    static void setSharedLogoTexture(unsigned int logoTextureId) {
        if (logoTextureId != 0) sharedLogoTextureId = logoTextureId;
    }

private:
    bool isDragging = false;
    int dragOffsetX = 0;
    int dragOffsetY = 0;

    inline static unsigned int sharedLogoTextureId = 0;
};